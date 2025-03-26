"""Configuration settings for the seasonal color classifier project."""

import os
from pathlib import Path
import torch

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "dataset/images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VIZ_DIR = os.path.join(BASE_DIR, "viz")

# Ensure viz directory exists
os.makedirs(VIZ_DIR, exist_ok=True)

# Dataset parameters
SEASONS = ["primavera", "estate", "autunno", "inverno"]
SUBTYPES = {
    "primavera": ["bright", "light", "warm"],
    "estate": ["cool", "light", "soft"],
    "autunno": ["deep", "soft", "warm"],
    "inverno": ["bright", "cool", "deep"]
}

# All 12 classes
ALL_CLASSES = [f"{season}_{subtype}" for season in SEASONS for subtype in SUBTYPES[season]]

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 250
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-4
EARLY_STOPPING_PATIENCE = 30

# Model parameters (default image size; may be overridden in main.py)
IMG_SIZE = 224
NUM_CLASSES = 12
FEATURE_EXTRACT = True
USE_PRETRAINED = True

# Augmentation parameters (no hue altering)
AUGMENTATION = True
RANDOM_ROTATION = 15
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
BRIGHTNESS_RANGE = (0.9, 1.1)
CONTRAST_RANGE = (0.9, 1.1)

# Normalization values (standard ImageNet)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Device configuration using torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fine-tuning parameters
FINE_TUNE = False        # Set to True to enable a fine-tuning stage
INITIAL_EPOCHS = 10      # Number of epochs to train classifier layers only before full fine-tuning
FINE_TUNE_LR = 5e-5

# -------------------------
# Hyperparameter Optimization Ranges
# -------------------------
HPO_RANGES = {
    "learning_rate": (1e-5, 1e-2),      # (min, max), log-scale
    "weight_decay": (1e-5, 1e-1),       # (min, max), linear-scale
    "fine_tune_lr": (1e-6, 1e-3),       # (min, max), log-scale
    "batch_size_options": [4,8,16]      # discrete choices
}

# Number of trials for hyperparameter optimization
HPO_TRIALS = 15

# Maximum epochs for each trial during HPO
HPO_MAX_EPOCHS = 200

