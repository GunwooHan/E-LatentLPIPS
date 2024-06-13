import argparse
import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR

from data.bapps import BAPPSDataModule
from e_latent_lpips import e_latent_lpips
from e_latent_lpips.e_latent_lpips import BCERankingLoss, LPIPS, CosineAnnealingWarmUpRestarts

import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionPipeline
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

from e_latent_lpips import e_latent_lpips

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--wandb', type=bool, default=False)
parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_scheduler', type=str, default="step")
parser.add_argument('--swa_lr', type=float, default=1e-4)

parser.add_argument('--target_model_path', type=str, default='checkpoints/vgg_scratch.ckpt')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/distillation')

parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--dataset_mode', type=str, default='2afc')
parser.add_argument('--train_dataset_dir', type=str, default=['train/traditional', 'train/cnn', 'train/mix'])
parser.add_argument('--val_dataset_dir', type=str,
                    default=['val/traditional', 'val/cnn', 'val/deblur', 'val/frameinterp', 'val/color',
                             'val/superres'])

args = parser.parse_args()


class DistillLPIPSModule(pl.LightningModule):
    def __init__(self, args):
        super(DistillLPIPSModule, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.teacher_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.teacher_model.eval()

        self.student_model = LPIPS(
            pretrained=True,
            spatial=False,
            pnet_rand=getattr(args, 'pnet_rand', False),
            pnet_tune=getattr(args, 'pnet_tune', False),
            use_dropout=True,
            latent_mode=True
        )

        self.rank_loss = BCERankingLoss()
        self.distillation_loss = nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.vae = pipe.vae
        self.vae = self.vae.to(device)

        self.psnr = PeakSignalNoiseRatio()
        self.model_trainalbe_set()

    def model_trainalbe_set(self):
        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad = False

    def training_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch

        # teacher
        teacher_d0s = self.teacher_model(ref_img, p0_img)
        teacher_d1s = self.teacher_model(ref_img, p1_img)
        teacher_gts = judge_img
        teacher_scores = (teacher_d0s < teacher_d1s) * (1. - teacher_gts) + (
                teacher_d1s < teacher_d0s) * teacher_gts + (teacher_d1s == teacher_d0s) * .5

        # student
        student_d0s = self.student_model(ref_img, p0_img)
        student_d1s = self.student_model(ref_img, p1_img)
        student_gts = judge_img
        student_scores = (student_d0s < student_d1s) * (1. - student_gts) + (
                student_d1s < student_d0s) * student_gts + (student_d1s == student_d0s) * .5
        student_rank_loss = self.rank_loss(student_d0s, student_d1s, student_gts)
        student_distillation_loss = self.distillation_loss(teacher_d0s, student_d0s) + self.distillation_loss(
            teacher_d1s, student_d1s)

        student_total_loss = student_rank_loss + student_distillation_loss

        self.log("train_teacher_rank_score", teacher_scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_student_distillation_score", student_distillation_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log("train_student_rank_loss", student_rank_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_student_score", student_scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_student_total_loss", student_total_loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        return student_total_loss

    def validation_step(self, batch: Any, idx: int):
        p0_img, p1_img, ref_img, judge_img = batch

        # teacher
        teacher_d0s = self.teacher_model(ref_img, p0_img)
        teacher_d1s = self.teacher_model(ref_img, p1_img)
        teacher_gts = judge_img
        teacher_scores = (teacher_d0s < teacher_d1s) * (1. - teacher_gts) + (
                teacher_d1s < teacher_d0s) * teacher_gts + (teacher_d1s == teacher_d0s) * .5

        # student
        student_d0s = self.student_model(ref_img, p0_img)
        student_d1s = self.student_model(ref_img, p1_img)
        student_gts = judge_img
        student_scores = (student_d0s < student_d1s) * (1. - student_gts) + (
                student_d1s < student_d0s) * student_gts + (student_d1s == student_d0s) * .5
        student_rank_loss = self.rank_loss(student_d0s, student_d1s, student_gts)
        student_distillation_loss = self.distillation_loss(teacher_d0s, student_d0s) + self.distillation_loss(
            teacher_d1s, student_d1s)

        student_total_loss = student_rank_loss + student_distillation_loss

        self.log("val_teacher_rank_score", teacher_scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_student_distillation_score", student_distillation_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log("val_student_rank_loss", student_rank_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_student_score", student_scores.mean() * 100, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_student_total_loss", student_total_loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True)

    #
    # def test_step(self, batch: Any, idx: int):
    #     p0_img, p1_img, ref_img, judge_img = batch
    #     d0s = self.model(ref_img, p0_img)
    #     d1s = self.model(ref_img, p1_img)
    #     gts = judge_img
    #     scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
    #     self.log("test_score", scores.mean() * 100, on_epoch=True, prog_bar=True)

    def on_train_batch_end(self, *args: Any):
        for module in self.student_model.lins.modules():
            if (hasattr(module, 'weight') and module.kernel_size == (1, 1)):
                module.weight.data = torch.clamp(module.weight.data, min=0.01)

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
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.001, T_up=5,
                                                      gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.args.gamma,
                                          patience=self.args.patience)
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


if __name__ == '__main__':
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        monitor='val_student_score',
        mode='max',
        filename='{val_student_score:.2f}',
        save_top_k=3
    )

    swa_callback = StochasticWeightAveraging(swa_lrs=args.swa_lr)
    lr_callback = LearningRateMonitor(logging_interval='step')

    if args.wandb:
        tag = []
        tag += args.train_dataset_dir
        tag += args.val_dataset_dir

        wandb_logger = WandbLogger(project='E-LatentLPIPS_distill', tags=tag)

    model = DistillLPIPSModule(args)

    dm = BAPPSDataModule(
        data_dir=args.data_dir,
        dataset_mode=args.dataset_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        args=args
    )

    trainer = pl.Trainer(
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, swa_callback, lr_callback],
        logger=wandb_logger if args.wandb else None,
    )
    trainer.fit(model, dm)
