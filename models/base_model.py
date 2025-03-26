"""Base model class for the seasonal color classifier."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Abstract base class for all model architectures."""
    
    def __init__(self, num_classes: int = 12):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @classmethod
    @abstractmethod
    def build_model(cls, **kwargs) -> 'BaseModel':
        pass
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

class ModelFactory:
    """Factory class for creating model instances using timm."""
    
    @staticmethod
    def get_model(model_name: str, **kwargs) -> 'BaseModel':
        from .cnn_models import TimmModel
        # If the provided model_name does not contain a dot, try to complete it.
        if '.' not in model_name:
            if model_name.startswith("vit_base_patch16_clip_224"):
                arch_name = "vit_base_patch16_clip_224.openai_ft_in12k_in1k"
            else:
                arch_name = model_name
        else:
            arch_name = model_name
        kwargs["arch_name"] = arch_name
        return TimmModel.build_model(**kwargs)
