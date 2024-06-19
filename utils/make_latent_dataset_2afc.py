import argparse
import glob
import os
import shutil

import torch
import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision import transforms

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
vae = pipe.vae
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
    for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.npy']:
        files += glob.glob(os.path.join(input_dir, "**", "*" + extension), recursive=True)

    num_batches = len(files) // batch_size + (0 if len(files) % batch_size == 0 else 1)

    for i in tqdm.tqdm(range(num_batches)):
        batch_files = files[i * batch_size: (i + 1) * batch_size]

        npy_files = [file for file in batch_files if file.endswith('.npy')]
        image_files = [file for file in batch_files if not file.endswith('.npy')]

        # 복사 작업
        for file in npy_files:
            output_file = file.replace(input_dir, output_dir)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            shutil.copyfile(file, output_file)

        if image_files:
            # preprocess image
            image_tensors = preprocess_image(image_files)

            # encode image
            latents = encode_image(image_tensors)

            # save encoded latents
            for file, latent in zip(image_files, latents):
                output_file = file.replace(os.path.splitext(file)[1], '.pt').replace(input_dir, output_dir)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                torch.save(latent.cpu(), output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='dataset/2afc')
    parser.add_argument('--output_dir', type=str, default='dataset/latent_2afc')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # 이미지 처리
    process_images(args.input_dir, args.output_dir, batch_size=args.batch_size)
