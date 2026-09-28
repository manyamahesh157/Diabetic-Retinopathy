"""
sanjeevani_ai.models.sanjeevani_model
-------------------------------------
Integrated Sanjeevani-AI Multi-Task Concept Bottleneck Network.

Full Pipeline:
Fundus Image -> EfficientNet Backbone -> CBAM Attention -> Concept Bottleneck -> Severity Head.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM
from .backbone import RetinalBackbone
from .concept_bottleneck import ConceptBottleneck, CLINICAL_CONCEPTS
from .severity_head import SeverityGradingHead, ICDR_GRADES


class SanjeevaniModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout_p: float = 0.30,
        num_classes: int = 5,
        concepts: list = CLINICAL_CONCEPTS,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.concepts = concepts
        self.num_concepts = len(concepts)

        # 1. Feature Extractor Backbone
        self.backbone = RetinalBackbone(backbone_name=backbone_name, pretrained=pretrained)
        in_features = self.backbone.num_features

        # 2. CBAM Attention Module
        self.attention = CBAM(planes=in_features)

        # 3. Global Pooling and Regularization
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)

        # 4. Concept Bottleneck
        self.concept_bottleneck = ConceptBottleneck(in_features=in_features, concepts=concepts)

        # 5. Severity Grading Head (Operates on Concept Vector)
        self.severity_head = SeverityGradingHead(num_concepts=self.num_concepts, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass from input fundus image to concepts and severity grades.
        """
        # Feature Extraction
        feats = self.backbone(x)            # (B, C, H, W)
        attended_feats = self.attention(feats)  # (B, C, H, W)

        # Global Pooling
        z = self.pool(attended_feats).flatten(1)  # (B, C)
        z = self.dropout(z)

        # Concept Bottleneck Extraction
        concept_dict = self.concept_bottleneck(z)
        concept_vector = concept_dict["concept_vector"]  # (B, K)

        # Severity Grading Head
        logits, attributions = self.severity_head(concept_vector)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "concept_vector": concept_vector,
            "concept_probs": concept_dict["concept_probs"],
            "concept_severities": concept_dict["concept_severities"],
            "concept_attributions": attributions,
            "spatial_features": attended_feats,
        }

    def forward_from_concepts(self, concept_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direct inference from concept vector to severity logits.
        Used by the Concept Counterfactual optimizer to solve:
           argmin_{delta c} ||delta c||^2 s.t. g(c + delta c) = target_grade
        in microseconds without re-evaluating the vision backbone.
        """
        logits, attributions = self.severity_head(concept_vector)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def get_attention_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature map after CBAM for Grad-CAM++ hook."""
        feats = self.backbone(x)
        return self.attention(feats)
