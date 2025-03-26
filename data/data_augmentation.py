"""Data augmentation techniques for the seasonal color classifier."""

import torch
from torchvision import transforms
from PIL import Image

from ..config import IMG_SIZE, RANDOM_ROTATION, HORIZONTAL_FLIP, VERTICAL_FLIP, BRIGHTNESS_RANGE, CONTRAST_RANGE, NORM_MEAN, NORM_STD

class ColorAugmentation:
    """Color augmentation techniques for seasonal color analysis."""
    
    @staticmethod
    def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5 if HORIZONTAL_FLIP else 0),
            transforms.RandomVerticalFlip(p=0.5 if VERTICAL_FLIP else 0),
            transforms.RandomRotation(RANDOM_ROTATION),
            transforms.ColorJitter(
                brightness=BRIGHTNESS_RANGE, 
                contrast=CONTRAST_RANGE, 
                saturation=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1),
            transforms.RandomApply([
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
            ], p=0.2),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    
    @staticmethod
    def get_test_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
