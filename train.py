import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from e_latent_lpips import e_latent_lpips
from data.bapps import BAPPSDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    # parser.add_argument('--optimizer', type=str, default='adamp')
    # parser.add_argument('--scheduler', type=str, default='reducelr')
    parser.add_argument('--wandb', type=bool, default=True)

    # parser.add_argument('--crop_image_size', type=int, default=512)
    # parser.add_argument('--ShiftScaleRotateMode', type=int, default=4)
    # parser.add_argument('--ShiftScaleRotate', type=float, default=0.2)
    # parser.add_argument('--HorizontalFlip', type=float, default=0.2)
    # parser.add_argument('--VerticalFlip', type=float, default=0.2)

    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--train_trunk', type=bool, default=False)
    parser.add_argument('--ensemble_model', type=bool, default=False)
    parser.add_argument('--latent_mode', type=bool, default=False)

    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--dataset_mode', type=str, default='2afc')
    parser.add_argument('--train_dataset_dir', type=str, default=['train/traditional', 'train/cnn', 'train/mix'])
    parser.add_argument('--val_dataset_dir', type=str, default=['val/traditional', 'val/cnn'])

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        monitor='val/score',
        mode='max',
        filename=f'{args.model}-' + '{epoch:02d}-{val/score:.2f}',
        save_top_k=3
    )

    if args.wandb:
        wandb_logger = WandbLogger(project='E-LatentLPIPS')

    model = e_latent_lpips.LPIPSModule(args)

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
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger if args.wandb else None
    )
    trainer.fit(model, dm)