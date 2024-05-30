from __future__ import absolute_import

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from .pretrained_networks import LatentVGG16, VGG16


class LPIPSModule(pl.LightningModule):
    def __init__(self, args: Any = None) -> None:
        super(LPIPSModule, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.model = LPIPS(
            pretrained=True,
            spatial=False,
            pnet_rand=getattr(args, 'pnet_rand', False),
            pnet_tune=getattr(args, 'pnet_tune', False),
            use_dropout=True,
            latent_mode=getattr(args, 'latent_mode', False)
        )

        self.rank_loss = BCERankingLoss()

    def training_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch
        d0s = self.model(ref_img, p0_img)
        d1s = self.model(ref_img, p1_img)
        gts = judge_img
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        self.log("train/score", scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = self.rank_loss(d0s, d1s, gts)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch
        d0s = self.model(ref_img, p0_img)
        d1s = self.model(ref_img, p1_img)
        gts = judge_img
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        self.log("val/score", scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = self.rank_loss(d0s, d1s, gts)
        self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def test_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch
        d0s = self.model(ref_img, p0_img)
        d1s = self.model(ref_img, p1_img)
        gts = judge_img
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        self.log("test/score", scores.mean() * 100, on_epoch=True, prog_bar=True)

    def on_train_batch_end(self, *args: Any):
        for module in self.model.lins.modules():
            if (hasattr(module, 'weight') and module.kernel_size == (1, 1)):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999))
        return optimizer

    def load_checkpoint(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    def forward(self, x0, x1, normalize=False):
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        x1 = (x1 - x1.min()) / (x1.max() - x1.min())
        x0 = x0 * 2 - 1
        x1 = x1 * 2 - 1
        return self.model(x0, x1)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIPS(nn.Module):
    def __init__(self,
                 pretrained=True,
                 spatial=False,
                 pnet_rand=False,
                 pnet_tune=False,
                 use_dropout=True,
                 latent_mode=False,
                 ):
        super(LPIPS, self).__init__()

        self.spatial = spatial
        self.latent_mode = latent_mode
        self.scaling_layer = ScalingLayer()

        self.channels = [64, 128, 256, 512, 512]

        if not latent_mode:
            self.net = VGG16(pretrained=pretrained)
        else:
            self.net = LatentVGG16(pretrained=pretrained)

        self.lin0 = NetLinLayer(self.channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.channels[4], use_dropout=use_dropout)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        if not self.latent_mode:
            in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        else:
            in0_input, in1_input = in0, in1

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        # normalize subtract
        for kk in range(len(self.channels)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.spatial:
            res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(len(self.channels))]
        else:
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(len(self.channels))]

        val = 0
        for l in range(len(self.channels)):
            val += res[l]

        if (retPerLayer):
            return (val, res)
        else:
            return val


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
        # for latent mode
        # self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188, -.030])[None, :, None, None])
        # self.register_buffer('scale', torch.Tensor([.458, .448, .450, .458])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        self.model = nn.Sequential(
            nn.Dropout() if use_dropout else nn.Identity,
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.model(x)


class Dist2LogitLayer(nn.Module):
    def __init__(self, channels=32):
        super(Dist2LogitLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, channels, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, channels=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(channels=channels)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = judge
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)
