import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_transforms(args=None):
    transform = []

    if not args.latent_mode:
        transform.append(A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if not args.latent_mode:
        transform.append(A.Resize(64, 64))

    transform.append(ToTensorV2())

    return A.Compose(transform,
                     additional_targets={
                         'image0': 'image',
                         'image1': 'image',
                     })
