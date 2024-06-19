import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from e_latent_lpips import e_latent_lpips
from data.bapps import BAPPSDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_scheduler', type=str, default="step")
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--t_max', type=int, default=5)
    parser.add_argument('--t_mult', type=int, default=5)
    parser.add_argument('--swa_lr', type=float, default=1e-4)

    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--latent_mode', type=bool, default=False)

    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--dataset_mode', type=str, default='2afc')
    parser.add_argument('--train_dataset_dir', type=str, default=['train/traditional', 'train/cnn', 'train/mix'])
    parser.add_argument('--val_dataset_dir', type=str,
                        default=['val/traditional', 'val/cnn', 'val/deblur', 'val/frameinterp', 'val/color',
                                 'val/superres'])

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        monitor='val_score',
        mode='max',
        filename=f'{args.model}-' + '{epoch:02d}-{val_score:.2f}',
        save_top_k=3
    )

    swa_callback = StochasticWeightAveraging(swa_lrs=args.swa_lr)
    lr_callback = LearningRateMonitor(logging_interval='step')

    if args.wandb:
        tag = []
        tag += args.train_dataset_dir
        tag += args.val_dataset_dir

        wandb_logger = WandbLogger(project='E-LatentLPIPS', tags=tag)

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
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, swa_callback, lr_callback],
        logger=wandb_logger if args.wandb else None,
    )
    trainer.fit(model, dm)
