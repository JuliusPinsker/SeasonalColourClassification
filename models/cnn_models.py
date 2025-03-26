"""State-of-the-art model implementations for the seasonal color classifier using timm.

Supported architectures:
    - hgnetv2_b5.ssld_stage2_ft_in1k
    - vit_base_patch16_clip_224.openai_ft_in12k_in1k
    - tf_efficientnetv2_l.in21k_ft_in1k
    - hgnetv2_b5.ssld_stage1_in22k_in1k
    - hgnet_base.ssld_in1k
    - coatnet_2_rw_224.sw_in12k_ft_in1k
    - convformer_m36.sail_in22k_ft_in1k
    - maxvit_base_tf_512.in1k
    - tf_efficientnetv2_xl.in21k_ft_in1k
    - convnextv2_huge.fcmae_ft_in1k
    - vit_base_patch8_224.augreg2_in21k_ft_in1k
    - vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k
"""

import timm
import torch.nn as nn
from .base_model import BaseModel
from ..config import NUM_CLASSES, USE_PRETRAINED

class TimmModel(BaseModel):
    """A timm model wrapper for seasonal color classification."""
    
    def __init__(self, model_name: str, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        self.model = timm.create_model(model_name, pretrained=USE_PRETRAINED, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    @classmethod
    def build_model(cls, arch_name: str, **kwargs) -> 'TimmModel':
        return cls(model_name=arch_name, **kwargs)
