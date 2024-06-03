import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms

def create_train_transforms(args=None):
    if not args.latent_mode:
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(64),
        ])


def create_valid_transforms(args=None):
    if not args.latent_mode:
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(64),
        ])

# def create_train_transforms(args=None):
#     transform = []
#
#     if not args.latent_mode:
#         transform.append(A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#
#     if not args.latent_mode:
#         transform.append(A.Resize(64, 64))
#
#     # Blit
#     #     Horizontal Flip
#     #     90-degree rotation
#     #     integer translation
#     # Geometric
#     #     isotropic scaling
#     #     aribitary rotation
#     #     fractional translation
#     # Cutout
#     # Color
#     #     brightness
#     #     saturation
#     #     random contrast
#
#     if args.blit:
#         # Horizontal Flip, 90-degree rotation, integer translation
#         transform.append(A.HorizontalFlip(p=0.2))
#         transform.append(A.Rotate(limit=90, p=0.2))
#         transform.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.2))
#
#     if args.geometric:
#         # isotropic scaling, arbitrary rotation, fractional translation
#         transform.append(A.ShiftScaleRotate(p=0.2))
#
#     if args.cutout:
#         # cutout
#         # https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/?h=coarsedropout
#         transform.append(A.CoarseDropout(max_holes=8, p=0.2))
#
#     if args.color:
#         transform.append(A.RandomBrightnessContrast(p=0.2))
#         transform.append(A.HueSaturationValue(p=0.2))
#
#     transform.append(ToTensorV2())
#
#     return A.Compose(transform,
#                      additional_targets={
#                          'image0': 'image',
#                          'image1': 'image',
#                      })
#
#
# def create_valid_transforms(args=None):
#     transform = []
#
#     if not args.latent_mode:
#         transform.append(A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#
#     if not args.latent_mode:
#         transform.append(A.Resize(64, 64))
#
#     transform.append(ToTensorV2())
#
#     return A.Compose(transform,
#                      additional_targets={
#                          'image0': 'image',
#                          'image1': 'image',
#                      })
