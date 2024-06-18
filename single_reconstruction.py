import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DModel
from pytorch_lightning.loggers import WandbLogger
from torch import optim, nn
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

from e_latent_lpips import e_latent_lpips
from augmentation_test import BrightnessAdjust, ContrastAdjust
from augment import AugmentPipe

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_scheduler', type=str, default="constant")

# parser.add_argument('--blit', type=bool, default=False)
# parser.add_argument('--geom', type=bool, default=False)
# parser.add_argument('--color', type=bool, default=False)
# parser.add_argument('--filter', type=bool, default=False)
# parser.add_argument('--noise', type=bool, default=False)
# parser.add_argument('--cutout', type=bool, default=False)

parser.add_argument('--xflip', type=bool, default=True)
parser.add_argument('--rotate90', type=bool, default=True)
parser.add_argument('--xint', type=bool, default=True)
parser.add_argument('--scale', type=bool, default=True)
parser.add_argument('--rotate', type=bool, default=True)
parser.add_argument('--aniso', type=bool, default=True)
parser.add_argument('--xfrac', type=bool, default=True)
parser.add_argument('--brightness', type=bool, default=True)
parser.add_argument('--contrast', type=bool, default=True)
parser.add_argument('--lumaflip', type=bool, default=False)
parser.add_argument('--hue', type=bool, default=False)
parser.add_argument('--saturation', type=bool, default=True)
parser.add_argument('--imgfilter', type=bool, default=False)
parser.add_argument('--noise', type=bool, default=True)
parser.add_argument('--cutout', type=bool, default=True)

parser.add_argument('--brightness_std', type=float, default=0.2)
parser.add_argument('--contrast_std', type=float, default=0.5)
parser.add_argument('--scale_std', type=float, default=0.2)
parser.add_argument('--rotate_max', type=float, default=1)
parser.add_argument('--aniso_std', type=float, default=0.2)
parser.add_argument('--xfrac_std', type=float, default=0.125)
parser.add_argument('--saturation_std', type=float, default=1)
parser.add_argument('--noise_std', type=float, default=1)
parser.add_argument('--cutout_size', type=float, default=0.5)

parser.add_argument('--reconstruction_target', type=str, default='/dev/shm/single_reconstruction_sample.jpeg')
parser.add_argument('--lpips_model_path', type=str,
                    default='checkpoints/latent_2afc/vgg-epoch=85-val_score=63.95.ckpt')
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--iterations', type=int, default=100000)
parser.add_argument('--latent_mode', type=bool, default=True)
parser.add_argument('--baseline', type=bool, default=False)
parser.add_argument('--ensemble_mode', type=bool, default=True)
args = parser.parse_args()


# torch.set_float32_matmul_precision('medium' | 'high')

class SingleReconstruction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = UNet2DModel(
            in_channels=4,  # the number of input channels, 3 for RGB images
            out_channels=4,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 512, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )
        # self.model = UNet2DModel(
        #     in_channels=4 if args.latent_mode else 3,  # the number of input channels, 3 for RGB images
        #     out_channels=4 if args.latent_mode else 3,  # the number of output channels
        #     layers_per_block=2,  # how many ResNet layers to use per UNet block
        #     block_out_channels=(256, 512, 512) if args.latent_mode else (128, 128, 256, 256, 512, 512),
        #     down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D") if args.latent_mode else (
        #         "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        #     up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D") if args.latent_mode else (
        #         "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        # )
        # model_id = "CompVis/stable-diffusion-v1-5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = args.gamma
        self.step_size = args.step_size
        self.args = args
        self.ensemble_mode = args.ensemble_mode

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(device)
        self.vae = pipe.vae
        # self.model = pipe.unet
        self.timestamp = args.seed
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.latent_mode = args.latent_mode
        self.baseline = args.baseline

        if self.latent_mode:
            self.lpips = e_latent_lpips.LPIPSModule.load_from_checkpoint(args.lpips_model_path, args=args)
            self.lpips.eval()
        else:
            if self.baseline:
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
            else:
                self.lpips = e_latent_lpips.LPIPSModule.load_from_checkpoint(args.lpips_model_path, args=args)
            self.lpips.eval()
        self.psnr = PeakSignalNoiseRatio()
        self.model_trainalbe_set()
        self.encode_hidden_state = self.encode_text(
            "The image shows a majestic castle on a hill, reflected in a calm body of water, under a starry night sky with a full moon. The castle is made of tan stone, has multiple pointed towers, and a flag on the central tower. ")

        if self.ensemble_mode:
            self.ensemble_transform = self.create_ensemble_transform().to(device)

        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    def model_trainalbe_set(self):
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        for name, param in self.lpips.named_parameters():
            param.requires_grad = False

    def create_ensemble_transform(self):
        # transform = []
        # if self.args.flip:
        #     transform.append(transforms.RandomHorizontalFlip())
        # if self.args.rotation:
        #     transform.append(transforms.RandomRotation(degrees=[-90, 90],
        #                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR))
        # if self.args.translation:
        #     transform.append(transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)))
        # if self.args.cutout:
        #     transform.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
        # if self.args.resize:
        #     transform.append(transforms.RandomResizedCrop(size=64, scale=(0.5, 1.5), ratio=(0.5, 1.5)))
        # if self.args.bright:
        #     transform.append(BrightnessAdjust(1))
        # if self.args.contrast:
        #     transform.append(ContrastAdjust(1))
        # return transforms.Compose(transform)
        return AugmentPipe(
            xflip=1 if self.args.xflip else 0,
            rotate90=1 if self.args.rotate90 else 0,
            xint=1 if self.args.xint else 0,
            scale=1 if self.args.scale else 0,
            rotate=1 if self.args.rotate else 0,
            aniso=1 if self.args.aniso else 0,
            xfrac=1 if self.args.xfrac else 0,
            brightness=1 if self.args.brightness else 0,
            contrast=1 if self.args.contrast else 0,
            lumaflip=1 if self.args.lumaflip else 0,
            hue=1 if self.args.hue else 0,
            saturation=1 if self.args.saturation else 0,
            imgfilter=1 if self.args.imgfilter else 0,
            noise=1 if self.args.noise else 0,
            cutout=1 if self.args.cutout else 0,
            brightness_std=args.brightness_std,
            contrast_std=args.contrast_std,
            scale_std=args.scale_std,
            rotate_max=args.rotate_max,
            aniso_std=args.aniso_std,
            xfrac_std=args.xfrac_std,
            saturation_std=args.saturation_std,
            noise_std=args.noise_std,
            cutout_size=args.cutout_size,
        )

        # Isotropic scaling (uniform scaling)

        # # Brightness adjustment
        # brightness = transforms.ColorJitter(brightness=0.5)
        #
        # # Saturation adjustment
        # saturation = transforms.ColorJitter(saturation=0.5)
        #
        # # Random contrast
        # random_contrast = transforms.ColorJitter(contrast=0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.latent_mode and batch_idx == 0:
            self.scaled_y_encode = self.vae.encode(y).latent_dist.sample() * 0.18215

        if self.latent_mode:
            # y_hat = self.model(x, args.seed, self.encode_hidden_state).sample
            scaled_latent = x  # * 0.18215
            unet_output = self.model(scaled_latent, args.seed).sample

            # y = 2 * (y - y.min()) / (y.max() - y.min()) - 1
            # y_hat = 2 * (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min()) - 1
        else:
            x0 = self.vae.encode(x).latent_dist.sample()
            x1 = self.model(x0, args.seed).sample
            y_hat = self.vae.decode(x1).sample
            #
            # # Normalize y and y_hat to be in range -1 to 1
            # y = 2 * (y - y.min()) / (y.max() - y.min()) - 1
            # y_hat = 2 * (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min()) - 1

        if self.ensemble_mode:
            transformed_unet_outputs_list = []
            transformed_scaled_y_encode_list = []

            # for i in range(x.size(0)):
            #     transformed = self.ensemble_transform(torch.cat([unet_output[i].unsqueeze(0), scaled_y_encode[i].unsqueeze(0)], dim=0))
            #     transformed_unet_outputs_list.append(transformed[:1, ...])
            #     transformed_scaled_y_encode_list.append(transformed[1:, ...])

            transformed_unet_outputs, transformed_scaled_y_encode = self.ensemble_transform(unet_output,
                                                                                            self.scaled_y_encode)

            # transformed_unet_outputs = torch.cat(transformed_unet_outputs_list, dim=0)
            # transformed_scaled_y_encode = torch.cat(transformed_scaled_y_encode_list, dim=0)

            lpips_loss = self.lpips(
                2 * (transformed_unet_outputs - transformed_unet_outputs.min()) / (
                        transformed_unet_outputs.max() - transformed_unet_outputs.min()) - 1,
                2 * (transformed_scaled_y_encode - transformed_scaled_y_encode.min()) / (
                        transformed_scaled_y_encode.max() - transformed_scaled_y_encode.min()) - 1).mean()
        else:
            lpips_loss = self.lpips(2 * (unet_output - unet_output.min()) / (unet_output.max() - unet_output.min()) - 1,
                                    2 * (self.scaled_y_encode - self.scaled_y_encode.min()) / (
                                            self.scaled_y_encode.max() - self.scaled_y_encode.min()) - 1).mean()
        # y = (y - y.min()) / (y.max() - y.min()) * 2 - 1
        # y_hat = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min()) * 2 - 1
        # lpips_torch_loss = self.lpips_torch(y, y_hat)

        self.log("lpips_loss", lpips_loss, on_step=True, prog_bar=True)
        # self.log("lpips_torch_loss", lpips_torch_loss, on_step=True, prog_bar=True)

        if batch_idx % 5000 == 0:
            if args.latent_mode:
                with torch.no_grad():
                    y_hat = self.vae.decode(unet_output[:1] / 0.18215).sample.clamp(min=-1, max=1)
                    psnr_loss = self.psnr(y[:1], y_hat)
                    self.logger.log_image("reconstruction", [y_hat], step=self.global_step + 1)
                    lpips_distance = self.lpips_metric(y[:1], y_hat)
                self.log("PSNR", psnr_loss, on_step=True, prog_bar=True)
                self.log("LPIPS", lpips_distance, on_step=True, prog_bar=True)
            else:
                self.logger.log_image("reconstruction", [y_hat], step=self.global_step + 1)
                psnr_loss = self.psnr(y, y_hat)
                self.log("PSNR", psnr_loss, on_step=True, prog_bar=True)

        return lpips_loss

    def forward(self, x):
        pass

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
        elif self.args.optimizer == 'sparse_adam':
            optimizer = optim.SparseAdam(self.parameters(), lr=self.args.learning_rate)
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
        elif self.args.lr_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.args.gamma, patience=5)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.args.lr_scheduler}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'lpips_loss'  # This is necessary for ReduceLROnPlateau
        }

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()
        with torch.no_grad():
            encoder_hidden_state = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        return encoder_hidden_state


class SingleReconstructionDataModule(pl.LightningModule):
    def __init__(self, target_image_path, num_workers: int = 0, latent_mode=False, batch_size=1):
        super().__init__()
        self.num_workers = num_workers
        self.target_image_path = target_image_path
        self.transform = self.create_transforms()
        self.batch_size = batch_size

        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.vae = self.pipe.vae
        self.vae.eval()

        image = Image.open(self.target_image_path)
        target_image = self.transform(image)
        # if latent_mode:
        #     with torch.no_grad():
        #         target_image = self.vae.encode(target_image.unsqueeze(0)).latent_dist.sample()  # Encode image
        #         # target_image = target_image * 0.18215  # Scaling factor from VAE
        #         target_image = target_image.squeeze(0)
        noise_image = torch.randn(4, 64, 64)

        self.dataset = SingleReconstructionDataset(noise_image, target_image)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        return 1000000000

    def __getitem__(self, idx):
        return self.noise_image, self.target_image


class ContrastAdjust(nn.Module):
    def __init__(self, contrast_factor=0):
        super().__init__()
        self.contrast_factor = contrast_factor

    def __call__(self, input_tensor):
        assert len(input_tensor.shape) == 4, "Input tensor must be 4-dimensional"

        batch_size, channels, height, width = input_tensor.shape

        # 4차원 입력 텐서에 대해 contrast 조정을 적용합니다.
        output_tensor = torch.empty_like(input_tensor)
        for i in range(batch_size):
            img = input_tensor[i]
            # 이미지를 각 채널별로 분리합니다.
            mean = img.mean([1, 2], keepdim=True)
            adjusted_img = (img - mean) * self.contrast_factor + mean
            output_tensor[i] = adjusted_img.clamp(0, 1)  # 값의 범위를 [0, 1]로 클램프합니다.

        return output_tensor


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    if args.wandb:
        wandb_logger = WandbLogger(
            project='E-LatentLPIPS_SingleReconstruction',
            tags=["latent"] if args.latent_mode else ["pixel"]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = SingleReconstructionDataModule(
        target_image_path=args.reconstruction_target,
        num_workers=args.num_workers,
        latent_mode=args.latent_mode,
        batch_size=args.batch_size
    )

    model = SingleReconstruction(args)
    # model.lpips.model.load_state_dict(torch.load("checkpoints/latest_net_.pth"))
    # model.lpips.rank_loss.net.load_state_dict(torch.load("checkpoints/latest_net_rank.pth"))

    trainer = pl.Trainer(
        devices=1,
        max_steps=args.iterations,
        logger=wandb_logger if args.wandb else None,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader())
