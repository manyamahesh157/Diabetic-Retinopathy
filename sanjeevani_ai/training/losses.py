"""
sanjeevani_ai.training.losses
-----------------------------
Clinically-Grounded Multi-Task Loss Formulation.

Addresses three core clinical realities:
1. Severe Class Imbalance: Grade 0 (No DR) represents 70-80% of real screening data.
   -> Addressed via Focal Loss (Lin et al., 2017) with gamma=2.0.
2. Ordinal Severity Structure: Misclassifying Grade 0 as Grade 4 is clinically catastrophic,
   whereas misclassifying Grade 1 as Grade 2 is a borderline adjacent ambiguity.
   -> Addressed via CORAL-style continuous expected-grade regression penalty.
3. Concept Supervision: Where lesion annotations exist (e.g. IDRiD), supervise concept heads
   with binary cross-entropy without penalizing unannotated samples.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1.0 - p_t) ** self.gamma) * logp_t

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)[targets]
            loss = loss * w

        return loss.mean()


class OrdinalSeverityPenalty(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        grades = torch.arange(self.num_classes, device=logits.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * grades, dim=-1)
        return F.mse_loss(expected_grade, targets.float())


class ConceptSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_targets: torch.Tensor,
        concept_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates BCE with logits only on samples and concepts that possess ground truth.
        concept_masks: Tensor (B, K) with 1 where annotation is verified, 0 where unknown.
        """
        bce = F.binary_cross_entropy_with_logits(concept_logits, concept_targets, reduction="none")
        if concept_masks is not None:
            bce = bce * concept_masks
            valid_count = concept_masks.sum()
            return bce.sum() / (valid_count + 1e-6) if valid_count > 0 else torch.tensor(0.0, device=concept_logits.device)
        return bce.mean()


class SanjeevaniMultiTaskLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        focal_gamma: float = 2.0,
        ordinal_weight: float = 0.35,
        concept_weight: float = 0.25,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
        self.ordinal = OrdinalSeverityPenalty(num_classes=num_classes)
        self.concept_loss = ConceptSupervisionLoss()
        self.ordinal_weight = ordinal_weight
        self.concept_weight = concept_weight

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        concept_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        logits = model_output["logits"]
        l_focal = self.focal(logits, targets)
        l_ordinal = self.ordinal(logits, targets)

        if concept_targets is not None:
            l_concept = self.concept_loss(
                model_output["concept_logits"], concept_targets, concept_masks
            )
        else:
            l_concept = torch.tensor(0.0, device=logits.device)

        total_loss = l_focal + self.ordinal_weight * l_ordinal + self.concept_weight * l_concept

        return {
            "total_loss": total_loss,
            "focal_loss": l_focal,
            "ordinal_loss": l_ordinal,
            "concept_loss": l_concept,
        }
