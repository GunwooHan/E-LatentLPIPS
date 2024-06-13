from __future__ import absolute_import

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR

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
        self.log("train_score", scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = self.rank_loss(d0s, d1s, gts)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch
        d0s = self.model(ref_img, p0_img)
        d1s = self.model(ref_img, p1_img)
        gts = judge_img
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        self.log("val_score", scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = self.rank_loss(d0s, d1s, gts)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def test_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch
        d0s = self.model(ref_img, p0_img)
        d1s = self.model(ref_img, p1_img)
        gts = judge_img
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        self.log("test_score", scores.mean() * 100, on_epoch=True, prog_bar=True)

    def on_train_batch_end(self, *args: Any):
        for module in self.model.lins.modules():
            if (hasattr(module, 'weight') and module.kernel_size == (1, 1)):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def configure_optimizers(self):
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamax':
            optimizer = optim.Adamax(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'nadam':
            optimizer = optim.NAdam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'radam':
            optimizer = optim.RAdam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'rprop':
            optimizer = optim.Rprop(self.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.args.optimizer}")


        if self.args.lr_scheduler == 'constant':
            return optimizer
        elif self.args.lr_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'exponential':
            scheduler = ExponentialLR(optimizer, gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'cosine_anneling':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.t_max, eta_min=1e-6)
        elif self.args.lr_scheduler == 'cosine_anneling_warmup_restarts':
            scheduler = CosineAnnealingWarmUpRestarts(optimizer,  T_0=10, T_mult=1, eta_max=0.001,  T_up=5, gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.args.gamma, patience=self.args.patience)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.args.lr_scheduler}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_total_loss'  # This is necessary for ReduceLROnPlateau
        }

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


import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmUpRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr