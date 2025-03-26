"""Evaluation metrics for the seasonal color classifier."""

from typing import Dict, List, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, class_names: List[str]) -> Dict[str, Union[float, np.ndarray, Dict]]:
    accuracy = accuracy_score(targets, predictions)
    precision_micro = precision_score(targets, predictions, average='micro', zero_division=0)
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall_micro = recall_score(targets, predictions, average='micro', zero_division=0)
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {'precision': precision_per_class[i], 'recall': recall_per_class[i], 'f1': f1_per_class[i]}
    cm = confusion_matrix(targets, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    season_map = {}
    for i, class_name in enumerate(class_names):
        season = class_name.split("_")[0] if "_" in class_name else class_name
        season_map[i] = season
    unique_seasons = sorted(set(season_map.values()))
    season_cm = np.zeros((len(unique_seasons), len(unique_seasons)))
    for i in range(len(targets)):
        true_class = targets[i]
        pred_class = predictions[i]
        true_season_idx = unique_seasons.index(season_map[true_class])
        pred_season_idx = unique_seasons.index(season_map[pred_class])
        season_cm[true_season_idx, pred_season_idx] += 1
    season_cm_normalized = season_cm.astype('float') / season_cm.sum(axis=1)[:, np.newaxis]
    
    metrics = {
        "accuracy": accuracy,
        "precision": {"micro": precision_micro, "macro": precision_macro, "weighted": precision_weighted},
        "recall": {"micro": recall_micro, "macro": recall_macro, "weighted": recall_weighted},
        "f1": {"micro": f1_micro, "macro": f1_macro, "weighted": f1_weighted},
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "season_confusion_matrix": season_cm,
        "season_confusion_matrix_normalized": season_cm_normalized,
        "seasons": unique_seasons
    }
    return metrics

class MetricsTracker:
    """Tracks evaluation metrics during training."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics_history = []
        self.best_metrics = None
        self.best_epoch = -1
    
    def update(self, metrics: Dict, epoch: int) -> bool:
        self.metrics_history.append((epoch, metrics))
        is_best = False
        if self.best_metrics is None or metrics["accuracy"] > self.best_metrics["accuracy"]:
            self.best_metrics = metrics
            self.best_epoch = epoch
            is_best = True
        return is_best
    
    def get_best_metrics(self) -> (Dict, int):
        return self.best_metrics, self.best_epoch
    
    def get_progress(self) -> Dict[str, list]:
        epochs = [item[0] for item in self.metrics_history]
        accuracies = [item[1]["accuracy"] for item in self.metrics_history]
        f1_macro = [item[1]["f1"]["macro"] for item in self.metrics_history]
        return {"epochs": epochs, "accuracy": accuracies, "f1_macro": f1_macro}
