"""
sanjeevani_ai.models.severity_head
----------------------------------
Severity Grading Head for ICDR Classification (Koh et al. Concept Bottleneck / Ordinal Mapping).

Maps interpretable concept vector c in [0, 1]^K into 5 International Clinical Diabetic
Retinopathy (ICDR) grades:
  Grade 0: No DR
  Grade 1: Mild NPDR
  Grade 2: Moderate NPDR
  Grade 3: Severe NPDR
  Grade 4: Proliferative DR

Because the severity head operates directly on the concept vector, clinicians and
judges can inspect the exact linear concept weights W_{j, k} that govern transitions
between severity stages.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

ICDR_GRADES = [
    "No DR (Grade 0)",
    "Mild NPDR (Grade 1)",
    "Moderate NPDR (Grade 2)",
    "Severe NPDR (Grade 3)",
    "Proliferative DR (Grade 4)",
]


class SeverityGradingHead(nn.Module):
    def __init__(self, num_concepts: int = 8, num_classes: int = 5):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # Linear concept attribution matrix (transparent mapping)
        self.linear_map = nn.Linear(num_concepts, num_classes)

        # Shallow non-linear refinement layer (captures inter-concept synergy, e.g. 4-2-1 rule)
        self.refinement = nn.Sequential(
            nn.Linear(num_concepts, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

        self._initialize_clinical_priors()

    def _initialize_clinical_priors(self):
        """
        Initializes weights with sound ophthalmological priors:
        - Grade 0: negative weights for all lesions.
        - Grade 1: positive for MA, negative for NV and severe hemorrhages.
        - Grade 2: positive for MA, Hemorrhages, Hard Exudates.
        - Grade 3: strong positive for extensive Hemorrhages, Venous Beading, IRMA.
        - Grade 4: dominant positive for Neovascularization.
        """
        with torch.no_grad():
            self.linear_map.weight.zero_()
            self.linear_map.bias.zero_()

            # Concept indices: 0:MA, 1:HEM, 2:HEX, 3:CWS, 4:VB, 5:IRMA, 6:NV, 7:MAC
            # Grade 0
            self.linear_map.weight[0, :] = -2.5
            self.linear_map.bias[0] = 1.8

            # Grade 1 (Mild): MA present
            self.linear_map.weight[1, 0] = 3.0   # MA
            self.linear_map.weight[1, 1] = -1.0  # Hem
            self.linear_map.weight[1, 6] = -4.0  # NV

            # Grade 2 (Moderate): MA + Hem + Exudates
            self.linear_map.weight[2, 0] = 2.0   # MA
            self.linear_map.weight[2, 1] = 2.5   # Hem
            self.linear_map.weight[2, 2] = 2.2   # Exudate
            self.linear_map.weight[2, 6] = -4.0  # NV

            # Grade 3 (Severe): Venous beading, extensive Hem, IRMA
            self.linear_map.weight[3, 1] = 3.5   # Hem
            self.linear_map.weight[3, 4] = 4.0   # VB
            self.linear_map.weight[3, 5] = 3.8   # IRMA
            self.linear_map.weight[3, 6] = -2.0  # NV

            # Grade 4 (PDR): Neovascularization is hallmark
            self.linear_map.weight[4, 6] = 6.0   # NV
            self.linear_map.weight[4, 1] = 2.0   # Hem

    def forward(self, concept_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            concept_vector: Tensor of shape (B, K)
        Returns:
            logits: Tensor of shape (B, 5)
            linear_attribution: Tensor of shape (B, 5, K) showing concept-to-grade contribution
        """
        # Linear transparent pathway
        linear_logits = self.linear_map(concept_vector)
        # Non-linear refinement
        ref_logits = self.refinement(concept_vector)

        logits = linear_logits + 0.3 * ref_logits

        # Concept-to-grade contribution: W_{j, k} * c_k
        batch_size = concept_vector.shape[0]
        # (5, K) * (B, 1, K) -> (B, 5, K)
        weights = self.linear_map.weight.unsqueeze(0)
        c_expanded = concept_vector.unsqueeze(1)
        attribution = weights * c_expanded

        return logits, attribution

    def compute_expected_grade(self, probs: torch.Tensor) -> torch.Tensor:
        """Computes continuous expected ordinal grade: sum(j * p_j)."""
        grades = torch.arange(self.num_classes, device=probs.device, dtype=torch.float32)
        return torch.sum(probs * grades, dim=-1)
