<div align="center">

# Unofficial Implementation E-LatentLPIPS(Ensembled-LatentLPIPS) of Diffusion2GAN
<img src="asset/LatentLPIPS_figure.jpg">

</div>

## News
- [24/07/11] release pretrained latent vgg network weights for E-Latent LPIPS
- [24/06/19] release Ensemble hyperparameter for E-Latent LPIPS
- [24/06/18] code release

## Comparison with Orginal Works

### Original Results
<div align="center">
<img src="asset/SingleRecon_figure.jpg">
</div>


### My Results
<div align="center">

<img src="asset/Single_Recon_my_figure.jpg">
</div>


It was implemented according to the contents of the paper, and the undisclosed details were implemented by referring to or speculating on the contents of the existing paper. The performance is lower than that of the paper, which is thought to be due to differences in the model architecture part. (Remove Maxpool layer position or 4th Channel scale/shift parameter) 


### Result Images by Augmentations Option
<div align="center">
<img src="asset/Aug_Result.jpg">
</div>

<div align="center">
<img src="asset/concatenated_output.gif">
</div>



I've reproduce it and the actual effect of color augmentation is thought to be insignificant
In the paper, Origianl Paper didn't use noise augmentation, but noise has a huge impact on the actual image recon quality
In the previous augmentation used in StyleGAN-ADA, augmentation such as luma flip also exists, but when added, it affected the reconstruction color and was excluded

<div align="center">
<img src="asset/Metrics.jpg">
</div>

The smaller the learning rate, the better the details of the image are reconstructed and the loss value is reduced, but not directly proportional to the PSNR




<div align="center">
<img src="asset/LearningRate_Comparison.jpg">
</div>

### Result Images by Augmentations Option Implemented by torchvision.transform

<div align="center">
<img src="asset/SingleRecon_StyleGAN-ADA.jpg">
</div>

The augmentation option used in StyleGAN-ADA was used as it was, but the reconstruction was not successful
Find an augmentation option that can reconstruct even detail features through Hyperparameter Optimization(4090 GPU 400 Hour+)

## How to use

### Install

```
pip install -r requirements.txt
```

ninja is a package that is installed for pytorch extension `upfirdn2d_plugin`, and you can install it optionally. There
is no problem with code execution even if you do not install it, but warning can appear

### Data Preparation

```bash
# BAPPS datset download
bash scripts/download_dataset.sh

# encode BAPPS dataset with runwayml/stable-diffusion-v1-5
# if you want to encode other VAE change huggingface url
python utils/make_latent_dataset_2afc.py --input_dir dataset/2afc --output_dir dataset/latent_2afc --batch_size 4
```

If you only want to download valset, run `scripts/download_dataset_valonly.sh` instead
of `bash scripts/download_dataset.sh`
These scripts are from the LPIPS repository, and so is the BAPPS dataset. See the repository for licenses related to
BAPPS dataset usage(https://github.com/richzhang/PerceptualSimilarity/tree/master)

By default, `make_latent_dataset_2afc.py` encode as `runwayml/stable-diffusion-v1-5`. If you want to use another VAE or
encoder, change the huggingface url

### checkpoints
you can download pretrained latent vgg netwotk this link (https://drive.google.com/file/d/1558700cub2hjAv-fXcyUGJUJBTrm5m3g/view?usp=sharing)

### Latent LPIPS Train
```
# train LPIPS
python train.py --dataset_mode 2afc

# train LatentLPIPS
python train.py --dataset_mode latent_2afc --latent_mode True
```

### Latent LPIPS test

```
# LPIPS
python test.py --model_path checkpoints/LatentLPIPS.ckpt --dataset_mode 2afc

LatentLPIPS
python test.py --model_path checkpoints/LatentLPIPS.ckpt --dataset_mode latent_2afc --latent_mode True
```

### Single Reconstruction Experiment

```
# load pretrained original LPIPS (https://github.com/richzhang/PerceptualSimilarity/tree/master)
python single_reconstruction.py --reconstruction_target single_reconstruction_sample.jpeg

# LPIPS
python single_reconstruction.py --reconstruction_target single_reconstruction_sample.jpeg

# LatentLPIPS
python single_reconstruction.py --reconstruction_target single_reconstruction_sample.jpeg --latent_mode

# E-LatentLPIPS
python single_reconstruction.py --reconstruction_target single_reconstruction_sample.jpeg --latent_mode --ensemble_mode

# Optimal option for E-LatentLPIPS
 python single_reconstruction.py --reconstruction_target single_reconstruction_sample.jpeg --latent_mode --ensemble_mode --xflip True --rotate90 True --xint True --xfrac True --scale True --rotate True --aniso True --brightness True --contrast True --saturation True --cutout True --noise True
```

## Citation

```
@article{kang2024diffusion2gan,
  author    = {Kang, Minguk and Zhang, Richard and Barnes, Connelly and Paris, Sylvain and Kwak, Suha and Park, Jaesik and Shechtman, Eli and Zhu, Jun-Yan and Park, Taesung},
  title     = {{Distilling Diffusion Models into Conditional GANs}},
  journal   = {arXiv preprint arXiv:2405.05967},
  year      = {2024},
}

@misc{2006.06676,
Author = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
Title = {Training Generative Adversarial Networks with Limited Data},
Year = {2020},
Eprint = {arXiv:2006.06676},

@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}

@misc{kettunen2019elpips,
      title={E-LPIPS: Robust Perceptual Image Similarity via Random Transformation Ensembles}, 
      author={Markus Kettunen and Erik Härkönen and Jaakko Lehtinen},
      year={2019},
      eprint={1906.03973},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}

```
