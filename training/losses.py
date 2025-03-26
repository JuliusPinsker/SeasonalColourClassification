"""Custom loss functions for seasonal color classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        loss = -focal_weight * torch.log(pt + 1e-12)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = inputs.size()
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smoothed_targets = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (num_classes - 1)
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(smoothed_targets * log_probs, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class SeasonWeightedLoss(nn.Module):
    """
    Custom loss function that penalizes misclassifications across seasons more heavily.
    """
    def __init__(self, classes: list, reduction: str = 'mean', base_weight: float = 1.0, across_weight: float = 2.0):
        super(SeasonWeightedLoss, self).__init__()
        self.reduction = reduction
        self.base_weight = base_weight
        self.across_weight = across_weight
        self.class_to_season = {}
        for i, class_name in enumerate(classes):
            season = class_name.split("_")[0] if "_" in class_name else class_name
            self.class_to_season[i] = season
        self.weight_matrix = self._create_weight_matrix(len(classes))
        
    def _create_weight_matrix(self, num_classes: int) -> torch.Tensor:
        weights = torch.ones(num_classes, num_classes) * self.across_weight
        weights.fill_diagonal_(0)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and self.class_to_season[i] == self.class_to_season[j]:
                    weights[i, j] = self.base_weight
        return weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        loss = F.nll_loss(log_probs, targets, reduction='none')
        _, preds = torch.max(inputs, 1)
        weights = torch.ones_like(loss)
        for i, (pred, target) in enumerate(zip(preds, targets)):
            if pred != target:
                weights[i] = self.weight_matrix[target, pred]
        weighted_loss = loss * weights
        if self.reduction == 'none':
            return weighted_loss
        elif self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
