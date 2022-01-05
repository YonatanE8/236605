from AdvancedDL import DATA_DIR
from torchvision.datasets import ImageFolder
from AdvancedDL.src.data.data_transforms import TwoCropsTransform, GaussianBlur

import os
import torchvision.transforms as transforms

CROP_SIZE = 300
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

IMAGENETTE_TRAIN_DIR = os.path.join(DATA_DIR, 'Imagenette', 'train')
self_training_transform = TwoCropsTransform(
    base_transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(CROP_SIZE, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
)
training_transforms = transforms.Compose(
    [
        transforms.Resize(CROP_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        normalize,
    ]
)

IMAGENETTE_VAL_DIR = os.path.join(DATA_DIR, 'Imagenette', 'val')
validation_transforms = transforms.Compose(
    [
        transforms.Resize(CROP_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        normalize,
    ]
)

imagenette_self_train_ds = ImageFolder(
    root=IMAGENETTE_TRAIN_DIR,
    transform=self_training_transform,
)
imagenette_train_ds = ImageFolder(
    root=IMAGENETTE_TRAIN_DIR,
    transform=training_transforms,
)
imagenette_val_ds = ImageFolder(
    root=IMAGENETTE_VAL_DIR,
    transform=validation_transforms,
)
