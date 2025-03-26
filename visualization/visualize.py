"""Visualization functions for the seasonal color classifier."""

from typing import Dict, List, Tuple, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from PIL import Image

from ..config import VIZ_DIR, ALL_CLASSES

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history["train_acc"], label="Training Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")
    plt.close()

def plot_learning_rate(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    if "lr" not in history:
        print("Learning rate not found in history")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning rate plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool = True, title: str = "Confusion Matrix", save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)) -> None:
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")
    plt.close()

def plot_prediction_samples(model: torch.nn.Module, images: List[torch.Tensor], true_labels: List[int], class_names: List[str], device: str = "cpu", num_samples: int = 16, save_path: Optional[str] = None) -> None:
    model.eval()
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image = images[idx].unsqueeze(0).to(device)
            true_label = true_labels[idx]
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.item()
            img = image.cpu() * std + mean
            img = img.squeeze(0).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            title = f"True: {class_names[true_label]}\nPred: {class_names[predicted]}"
            color = "green" if predicted == true_label else "red"
            ax.imshow(img)
            ax.set_title(title, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction samples plot saved to {save_path}")
    plt.close()

def plot_class_distribution(train_labels: List[int], test_labels: List[int], class_names: List[str], save_path: Optional[str] = None) -> None:
    import pandas as pd
    train_counts = np.bincount(train_labels, minlength=len(class_names))
    test_counts = np.bincount(test_labels, minlength=len(class_names))
    df = pd.DataFrame({
        'Class': class_names,
        'Training': train_counts,
        'Testing': test_counts
    })
    df_melted = pd.melt(df, id_vars=['Class'], value_vars=['Training', 'Testing'], var_name='Dataset', value_name='Count')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', hue='Dataset', data=df_melted)
    plt.xticks(rotation=45, ha='right')
    plt.title('Class Distribution in Training and Test Sets')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Class distribution plot saved to {save_path}")
    plt.close()

def plot_season_metrics(metrics: Dict, save_path: Optional[str] = None) -> None:
    season_cm = metrics.get("season_confusion_matrix_normalized")
    seasons = metrics.get("seasons")
    if season_cm is None or seasons is None:
        print("Season metrics not found.")
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(season_cm, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=seasons, yticklabels=seasons)
    plt.xlabel("Predicted Season")
    plt.ylabel("True Season")
    plt.title("Normalized Season Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Season metrics plot saved to {save_path}")
    plt.close()
