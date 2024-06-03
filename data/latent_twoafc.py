import glob
import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LatentTwoAFCDataset(Dataset):
    def __init__(self, data_dir, dataset_dirs, transform=None):
        self.ref_paths = []
        self.p0_paths = []
        self.p1_paths = []
        self.judge_paths = []

        for dataset_dir in dataset_dirs:
            self.ref_paths += sorted(glob.glob(os.path.join(data_dir, "latent_2afc", dataset_dir, "ref", "*")))
            self.p0_paths += sorted(glob.glob(os.path.join(data_dir, "latent_2afc", dataset_dir, "p0", "*")))
            self.p1_paths += sorted(glob.glob(os.path.join(data_dir, "latent_2afc", dataset_dir, "p1", "*")))
            self.judge_paths += sorted(glob.glob(os.path.join(data_dir, "latent_2afc", dataset_dir, "judge", "*")))

            path_lengths = [len(self.ref_paths), len(self.p0_paths), len(self.p1_paths), len(self.judge_paths)]
            if len(set(path_lengths)) != 1:
                msg = f"Paths do not have the same length. Length of ref_paths: {len(self.ref_paths)}, p0_paths: {len(self.p0_paths)}, p1_paths: {len(self.p1_paths)}, judge_paths: {len(self.judge_paths)}."
                raise ValueError(msg)

        self.transform = transform

    def __getitem__(self, idx):
        p0_img, p1_img, ref_img = self.apply_transform_with_seed(
            [self.p0_paths[idx], self.p1_paths[idx], self.ref_paths[idx]])

        judge_path = self.judge_paths[idx]
        judge_img = np.load(judge_path).reshape((1, 1, 1,))  # [0,1]

        judge_img = torch.FloatTensor(judge_img)

        # For Debug
        import torchvision.utils as vutils
        temp_tensor = [p0_img, p1_img, ref_img]
        vutils.save_image(temp_tensor, "output.png", nrow=3, normalize=True)

        return p0_img, p1_img, ref_img, judge_img

    def __len__(self):
        return len(self.p0_paths)

    def apply_transform_with_seed(self, image_paths):
        p0_img = torch.load(image_paths[0])
        p1_img = torch.load(image_paths[1])
        ref_img = torch.load(image_paths[2])

        p0_img = (p0_img - p0_img.min()) / (p0_img.max() - p0_img.min())
        p1_img = (p1_img - p1_img.min()) / (p1_img.max() - p1_img.min())
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())

        p0_img = self.transform(p0_img)
        p1_img = self.transform(p1_img)
        ref_img = self.transform(ref_img)

        p0_img = p0_img * 2 - 1
        p1_img = p1_img * 2 - 1
        ref_img = ref_img * 2 - 1
        return p0_img, p1_img, ref_img


if __name__ == '__main__':
    # Define your directory and dataset directories here
    directory = "../dataset"
    dataset_dirs = ['train/traditional', 'train/cnn',
                    'train/mix']  # List of dataset subdirectories within your directory

    from albumentations.pytorch import ToTensorV2
    import albumentations as A

    transform = [
        A.RandomRotate90(),
    ]

    dataset = LatentTwoAFCDataset(directory, dataset_dirs, A.Compose(transform,
                                                                     additional_targets={
                                                                         'image0': 'image',
                                                                         'image1': 'image',
                                                                     }))

    # Test the data loader by printing images from a batch
    for i, data in enumerate(dataset, 0):
        p0, p1, ref, judge = data
        print(p0.size(), p1.size(), ref.size(), judge.size())
        if i == 3:
            break
