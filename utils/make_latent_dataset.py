import glob
import os
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import tqdm

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
vae.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)


def preprocess_image(image_paths):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize([0.5], [0.5]),
    ])
    images = [transform(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    return torch.stack(images)  # Convert list of tensors to single tensor


def encode_image(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
    return latents


def process_images(input_dir, output_dir, batch_size=32):
    files = []
    for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
        files += glob.glob(os.path.join(input_dir, "**", "*" + extension), recursive=True)

    num_batches = len(files) // batch_size + (0 if len(files) % batch_size == 0 else 1)

    for i in tqdm.tqdm(range(num_batches)):
        batch_files = files[i * batch_size: (i + 1) * batch_size]

        for file in batch_files:
            os.makedirs(os.path.dirname(file.replace(input_dir, output_dir)), exist_ok=True)

        # preprocess image
        image_tensors = preprocess_image(batch_files)

        # encode image
        latents = encode_image(image_tensors)

        # save encoded latents
        for file, latent in zip(batch_files, latents):
            np.save(file.replace(os.path.splitext(file)[1], '.npy').replace(input_dir, output_dir),
                    latent.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='dataset/2afc/val/traditional')
    parser.add_argument('--output_dir', type=str, default='dataset/latent_2afc/val/traditional')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # 이미지 처리
    process_images(args.input_dir, args.output_dir, batch_size=args.batch_size)
