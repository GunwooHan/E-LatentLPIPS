import argparse

import pytorch_lightning as pl
import torch
from PIL import Image
from diffusers import UNet2DModel, AutoencoderKL
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity


from e_latent_lpips import e_latent_lpips

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--reconstruction_target', type=str, default='single_reconstruction_sample.jpeg')
parser.add_argument('--lpips_model_path', type=str, default='checkpoints/origin/vgg-epoch=38-val/score=75.74.ckpt')
parser.add_argument('--wandb', type=str, default=True)
parser.add_argument('--iterations', type=int, default=50000)
parser.add_argument('--latent_mode', type=bool, default=True)
args = parser.parse_args()


class SingleReconstruction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = UNet2DModel(
            in_channels=4 if args.latent_mode else 3,  # the number of input channels, 3 for RGB images
            out_channels=4 if args.latent_mode else 3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 512, 512) if args.latent_mode else (128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D") if args.latent_mode else (
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D") if args.latent_mode else (
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        self.lpips = e_latent_lpips.LPIPSModule.load_from_checkpoint(args.lpips_model_path, args=args)

        for n, layer in enumerate(self.lpips.parameters()):
            layer.requires_grad_(False)

        # self.lpips_torch = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

        self.psnr = PeakSignalNoiseRatio()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        self.vae.eval()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x, args.seed).sample

        lpips_loss = self.lpips(y, y_hat).flatten()
        psnr_loss = self.psnr(y, y_hat)
        # y = (y - y.min()) / (y.max() - y.min()) * 2 - 1
        # y_hat = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min()) * 2 - 1
        # lpips_torch_loss = self.lpips_torch(y, y_hat)

        self.log("lpips_loss", lpips_loss, on_step=True, prog_bar=True)
        # self.log("lpips_torch_loss", lpips_torch_loss, on_step=True, prog_bar=True)
        self.log("psnr_loss", psnr_loss, on_step=True, prog_bar=True)

        if self.global_step % 5000 == 0:
            if args.latent_mode:
                with torch.no_grad():
                    log_sample_image = self.vae.decode(y_hat).sample
                    self.logger.log_image("reconstruction", [log_sample_image], step=self.global_step+1)
            else:
                pass
        return lpips_loss

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999))
        return optimizer


class SingleReconstructionDataModule(pl.LightningModule):
    def __init__(self, target_image_path, num_workers: int = 0, latent_mode=False):
        super().__init__()
        self.num_workers = num_workers
        self.target_image_path = target_image_path
        self.transform = self.create_transforms()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        self.vae.eval()

        image = Image.open(self.target_image_path)
        target_image = self.transform(image)
        if latent_mode:
            with torch.no_grad():
                target_image = self.vae.encode(target_image.unsqueeze(0)).latent_dist.sample()  # Encode image
                # target_image = target_image * 0.18215  # Scaling factor from VAE
                target_image = target_image.squeeze(0)
        noise_image = torch.randn_like(target_image)

        self.dataset = SingleReconstructionDataset(noise_image, target_image)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def create_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return transform


class SingleReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, noise_image, target_image):
        self.noise_image = noise_image
        self.target_image = target_image

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        return self.noise_image, self.target_image


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    if args.wandb:
        wandb_logger = WandbLogger(
            project='E-LatentLPIPS SingleReconstruction',
            tags=["latent"] if args.latent_mode else ["pixel"]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = SingleReconstructionDataModule(
        target_image_path=args.reconstruction_target,
        num_workers=args.num_workers,
        latent_mode=args.latent_mode
    )

    model = SingleReconstruction(args)

    trainer = pl.Trainer(
        devices=1,
        max_steps=args.iterations,
        logger=wandb_logger if args.wandb else None
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader())
