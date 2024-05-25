import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from e_latent_lpips import e_latent_lpips
from data.bapps import BAPPSDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--model_path', type=str, default='checkpoints/vggepoch=08-val/score=81.17.ckpt')

    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--dataset_mode', type=str, default='2afc')
    parser.add_argument('--val_dataset_dir', type=str, nargs='+', default=['val/cnn'])
    parser.add_argument('--latent_mode', type=bool, default=False)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project='E-LatentLPIPS')

    pl.seed_everything(args.seed)

    if os.path.splitext(args.model_path)[1] in ['.pt', '.pth']:
        model = e_latent_lpips.LPIPSModule()
        model.load_checkpoint(args.model_path)
    elif os.path.splitext(args.model_path)[1] == '.ckpt':
        model = e_latent_lpips.LPIPSModule.load_from_checkpoint(args.model_path)
    else:
        model = e_latent_lpips.LPIPSModule()

    dm = BAPPSDataModule(
        data_dir=args.data_dir,
        dataset_mode=args.dataset_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_dataset_dir=args.val_dataset_dir,
        args=args
    )

    trainer = pl.Trainer(
        logger=wandb_logger
    )
    trainer.test(model, dm)
