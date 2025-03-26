"""Training logic for seasonal color classification models with Neptune integration.

This trainer performs a two-stage training when fine-tuning is enabled:
  1. Initial training for INITIAL_EPOCHS epochs (training only the classifier layers).
  2. Fine-tuning for the remaining epochs (unfreezing the entire network).

If fine-tuning is disabled, the whole network is trained for all epochs.
"""

import os
import time
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..models.base_model import BaseModel
from ..evaluation.metrics import calculate_metrics
from ..config import DEVICE, EARLY_STOPPING_PATIENCE, LEARNING_RATE, WEIGHT_DECAY, FINE_TUNE, INITIAL_EPOCHS, FINE_TUNE_LR

class Trainer:
    """Trainer class for the seasonal color classification model."""
    
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = DEVICE,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        save_dir: str = "checkpoints",
        neptune_run: Optional[object] = None,
        custom_learning_rate: float = None,
        custom_weight_decay: float = None,
        custom_fine_tune_lr: float = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        
        self.learning_rate = custom_learning_rate if custom_learning_rate is not None else LEARNING_RATE
        self.weight_decay = custom_weight_decay if custom_weight_decay is not None else WEIGHT_DECAY
        self.fine_tune_lr = custom_fine_tune_lr if custom_fine_tune_lr is not None else FINE_TUNE_LR

        self.device = device
        self.early_stop = early_stopping_patience
        self.save_dir = save_dir
        self.neptune_run = neptune_run
        self.model.to(self.device)
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.best_checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
        
        # For fine-tuning, initially freeze base layers (train classifier only)
        if FINE_TUNE:
            for name, param in self.model.named_parameters():
                if ("fc" in name) or ("classifier" in name) or ("head" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            # Train whole network
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({"loss": loss.item(), "acc": correct / total})
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), self.val_loader.dataset.classes)
        return val_loss, metrics["accuracy"], metrics
    
    def train(self, num_epochs: int = 50) -> Dict[str, list]:
        # Determine initial and fine-tuning epochs based on config and FINE_TUNE flag
        if FINE_TUNE:
            initial_epochs = INITIAL_EPOCHS
            fine_tune_epochs = num_epochs - INITIAL_EPOCHS
            print(f"Starting initial training for {initial_epochs} epochs (training classifier layers only)...")
        else:
            initial_epochs = num_epochs
            fine_tune_epochs = 0
            print(f"Starting training for {num_epochs} epochs (training whole network)...")
        
        start_time = time.time()
        current_epoch = 0

        # ----- Initial Training Phase -----
        for epoch in range(initial_epochs):
            current_epoch += 1
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, metrics = self.validate_epoch()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {current_epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            
            if self.neptune_run is not None:
                self.neptune_run["train/loss"].append(train_loss)
                self.neptune_run["train/accuracy"].append(train_acc)
                self.neptune_run["val/loss"].append(val_loss)
                self.neptune_run["val/accuracy"].append(val_acc)
                self.neptune_run["lr"].append(current_lr)
                self.neptune_run["eval/accuracy"].append(metrics["accuracy"])
                self.neptune_run["eval/precision/micro"].append(metrics["precision"]["micro"])
                self.neptune_run["eval/precision/macro"].append(metrics["precision"]["macro"])
                self.neptune_run["eval/precision/weighted"].append(metrics["precision"]["weighted"])
                self.neptune_run["eval/recall/micro"].append(metrics["recall"]["micro"])
                self.neptune_run["eval/recall/macro"].append(metrics["recall"]["macro"])
                self.neptune_run["eval/recall/weighted"].append(metrics["recall"]["weighted"])
                self.neptune_run["eval/f1/micro"].append(metrics["f1"]["micro"])
                self.neptune_run["eval/f1/macro"].append(metrics["f1"]["macro"])
                self.neptune_run["eval/f1/weighted"].append(metrics["f1"]["weighted"])
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = current_epoch - 1
                self.epochs_without_improvement = 0
                self.save_checkpoint(self.best_checkpoint_path)
                print(f"Saved best model (initial phase) with val_loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.early_stop:
                print(f"Early stopping triggered during initial training after {current_epoch} epochs")
                break
        
        # ----- Fine-Tuning Phase (if enabled) -----
        if FINE_TUNE and fine_tune_epochs > 0:
            print("Reloading best model checkpoint for fine-tuning (unfreezing entire network)...")
            self.load_checkpoint(self.best_checkpoint_path)
            # Unfreeze all layers for fine-tuning
            for param in self.model.parameters():
                param.requires_grad = True
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.fine_tune_lr,
                weight_decay=self.weight_decay
            )
            print(f"Starting fine-tuning for {fine_tune_epochs} epochs...")
            self.epochs_without_improvement = 0
            for ft_epoch in range(fine_tune_epochs):
                current_epoch += 1
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc, metrics = self.validate_epoch()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Fine-tune Epoch {ft_epoch+1}/{fine_tune_epochs} (Total Epoch {current_epoch}/{num_epochs}) - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
                
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_acc"].append(val_acc)
                self.history["lr"].append(current_lr)
                
                if self.neptune_run is not None:
                    self.neptune_run["train/loss"].append(train_loss)
                    self.neptune_run["train/accuracy"].append(train_acc)
                    self.neptune_run["val/loss"].append(val_loss)
                    self.neptune_run["val/accuracy"].append(val_acc)
                    self.neptune_run["lr"].append(current_lr)
                    self.neptune_run["eval/accuracy"].append(metrics["accuracy"])
                    self.neptune_run["eval/precision/micro"].append(metrics["precision"]["micro"])
                    self.neptune_run["eval/precision/macro"].append(metrics["precision"]["macro"])
                    self.neptune_run["eval/precision/weighted"].append(metrics["precision"]["weighted"])
                    self.neptune_run["eval/recall/micro"].append(metrics["recall"]["micro"])
                    self.neptune_run["eval/recall/macro"].append(metrics["recall"]["macro"])
                    self.neptune_run["eval/recall/weighted"].append(metrics["recall"]["weighted"])
                    self.neptune_run["eval/f1/micro"].append(metrics["f1"]["micro"])
                    self.neptune_run["eval/f1/macro"].append(metrics["f1"]["macro"])
                    self.neptune_run["eval/f1/weighted"].append(metrics["f1"]["weighted"])
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = current_epoch - 1
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(self.best_checkpoint_path)
                    print(f"Saved best model during fine-tuning with val_loss: {val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stop:
                    print(f"Early stopping triggered during fine-tuning after {ft_epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best epoch: {self.best_epoch+1} with val_loss: {self.best_val_loss:.4f}")
        return self.history
    
    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.best_epoch,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['epoch']
