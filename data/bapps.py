from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.transforms import create_train_transforms, create_valid_transforms
from .latent_twoafc import LatentTwoAFCDataset
from .twoafc import TwoAFCDataset


class BAPPSDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./dataset",
                 dataset_mode: str = '2afc',
                 batch_size: int = 32,
                 num_workers: int = 0,
                 train_dataset_dir: List = None,
                 val_dataset_dir: List = None,
                 args: dict = None,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir

        self.data_dir = data_dir
        self.train_transform = create_train_transforms(self.args)
        self.valid_transform = create_valid_transforms(self.args)

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        if stage == "fit":
            if self.dataset_mode == '2afc':
                self.bapps_train = TwoAFCDataset(self.data_dir, self.train_dataset_dir, self.train_transform)
                self.bapps_val = TwoAFCDataset(self.data_dir, self.val_dataset_dir, self.valid_transform)
            elif self.dataset_mode == 'jnd':
                raise f'{self.dataset_mode} Not implemented yet'
            elif self.dataset_mode == 'latent_2afc':
                self.bapps_train = LatentTwoAFCDataset(self.data_dir, self.train_dataset_dir, self.train_transform)
                self.bapps_val = LatentTwoAFCDataset(self.data_dir, self.val_dataset_dir, self.valid_transform)
            elif self.dataset_mode == 'latent_jnd':
                raise f'{self.dataset_mode} Not implemented yet'
            else:
                raise f'Invalid Dataset Name : {self.dataset_mode}'

            print(f'Train Dataset {self.dataset_mode} : {self.train_dataset_dir}')
            print(f'Validate Dataset {self.dataset_mode} : {self.val_dataset_dir}')
            print('Number of training samples: {}'.format(len(self.bapps_train)))
            print('Number of validating samples: {}'.format(len(self.bapps_val)))

        elif stage == "test":
            if self.dataset_mode == '2afc':
                self.bapps_test = TwoAFCDataset(self.data_dir, self.val_dataset_dir, self.valid_transform)
            elif self.dataset_mode == 'jnd':
                raise f'{self.dataset_mode} Not implemented yet'
            elif self.dataset_mode == 'latent_2afc':
                self.bapps_test = LatentTwoAFCDataset(self.data_dir, self.val_dataset_dir, self.valid_transform)
            elif self.dataset_mode == 'latent_jnd':
                raise f'{self.dataset_mode} Not implemented yet'
            else:
                raise f'Invalid Dataset Name : {self.dataset_mode}'

            print(f'Validate Dataset {self.dataset_mode} : {self.val_dataset_dir}')
            print(f'Number of validating samples: {len(self.bapps_test)}')

    def train_dataloader(self):
        return DataLoader(
            self.bapps_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.bapps_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.bapps_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False
        )

    def predict_dataloader(self):
        pass
