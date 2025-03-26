"""Data loading and preprocessing module for the seasonal color classifier."""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from ..config import SEASONS, SUBTYPES, IMG_SIZE, BATCH_SIZE, DEVICE, NORM_MEAN, NORM_STD, RANDOM_ROTATION, BRIGHTNESS_RANGE, CONTRAST_RANGE

class SeasonalColorDataset(Dataset):
    """Dataset for seasonal color classification."""
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        class_idx = 0
        for season in SEASONS:
            season_dir = os.path.join(self.root_dir, season)
            if any(os.path.isdir(os.path.join(season_dir, d)) for d in os.listdir(season_dir)):
                for subtype in SUBTYPES[season]:
                    subtype_dir = os.path.join(season_dir, subtype)
                    if os.path.exists(subtype_dir):
                        class_name = f"{season}_{subtype}"
                        self.classes.append(class_name)
                        self.class_to_idx[class_name] = class_idx
                        for img_name in os.listdir(subtype_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(subtype_dir, img_name)
                                self.samples.append((img_path, class_idx))
                        class_idx += 1
            else:
                class_name = season
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(season_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(season_dir, img_name)
                        self.samples.append((img_path, class_idx))
                class_idx += 1
        print(f"Loaded {len(self.samples)} images across {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(
    train_dir: str, 
    test_dir: str, 
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(RANDOM_ROTATION),
        transforms.ColorJitter(brightness=BRIGHTNESS_RANGE, contrast=CONTRAST_RANGE, saturation=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_dataset = SeasonalColorDataset(train_dir, transform=train_transform)
    test_dataset = SeasonalColorDataset(test_dir, transform=test_transform)
    
    pin = True if DEVICE == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    
    return train_loader, test_loader
