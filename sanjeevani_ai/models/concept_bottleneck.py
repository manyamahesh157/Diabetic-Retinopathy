"""
sanjeevani_ai.models.concept_bottleneck
---------------------------------------
Concept Bottleneck Layer (Koh et al., ICML 2020 / Medical Adaptation).

Instead of an end-to-end black box:
    Image -> CNN -> Logits -> Softmax
The architecture forces the network to route through an intermediate layer of
clinically validated ophthalmological concepts:
    Image -> Shared Features -> Concept Heads (c) -> Severity Grading Head (y)

This design ensures:
1. Every severity prediction is causally and inspectably grounded in specific lesion concepts.
2. Clinicians can view and verify intermediate lesion confidence scores.
3. Counterfactual reasoning can be performed directly in concept space.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

CLINICAL_CONCEPTS = [
    {
        "id": "microaneurysms",
        "name": "Microaneurysms",
        "category": "NPDR Early",
        "status": "IMPLEMENTED",
        "source": "IDRiD Lesion Segmentations",
    },
    {
        "id": "hemorrhages",
        "name": "Retinal Hemorrhages",
        "category": "NPDR Intermediate",
        "status": "IMPLEMENTED",
        "source": "IDRiD Lesion Segmentations",
    },
    {
        "id": "hard_exudates",
        "name": "Hard Exudates",
        "category": "Lipid Exudation",
        "status": "IMPLEMENTED",
        "source": "IDRiD Lesion Segmentations",
    },
    {
        "id": "cotton_wool_spots",
        "name": "Cotton-Wool Spots",
        "category": "Ischemia / Soft Exudates",
        "status": "IMPLEMENTED",
        "source": "IDRiD Lesion Segmentations",
    },
    {
        "id": "venous_beading",
        "name": "Venous Beading",
        "category": "Severe NPDR (4-2-1 Rule)",
        "status": "EXPERIMENTAL",
        "source": "Weakly-Supervised Clinical Prior",
    },
    {
        "id": "irma",
        "name": "IRMA",
        "category": "Severe NPDR Shunts",
        "status": "EXPERIMENTAL",
        "source": "Weakly-Supervised Clinical Prior",
    },
    {
        "id": "neovascularization",
        "name": "Neovascularization",
        "category": "Proliferative Hallmark",
        "status": "IMPLEMENTED",
        "source": "APTOS/IDRiD High-Grade Annotations",
    },
    {
        "id": "macular_edema_risk",
        "name": "Macular Involvement (CSME)",
        "category": "Sight-Threatening Edema",
        "status": "IMPLEMENTED",
        "source": "Foveal Proximity + Exudate Burden",
    },
]


class ConceptHead(nn.Module):
    """
    Dedicated MLP head for an individual clinical concept.
    Predicts:
    1. presence_logit (scalar logit for binary classification)
    2. severity_score (continuous scalar [0, 1] representing lesion burden/extent)
    """

    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 2),  # [0]: presence logit, [1]: severity score logit
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(z)
        presence_logit = out[:, 0]
        severity_score = torch.sigmoid(out[:, 1])
        return presence_logit, severity_score


class ConceptBottleneck(nn.Module):
    """
    Extracts K clinical concepts from pooled representation z.
    Produces:
    - concept_vector: Tensor of shape (B, K) with values in [0, 1]
    - concept_logits: Tensor of shape (B, K)
    - concept_severities: Tensor of shape (B, K) in [0, 1]
    """

    def __init__(self, in_features: int, concepts: List[Dict[str, Any]] = CLINICAL_CONCEPTS):
        super().__init__()
        self.concept_specs = concepts
        self.concept_ids = [c["id"] for c in concepts]
        self.num_concepts = len(concepts)

        self.heads = nn.ModuleDict({
            c["id"]: ConceptHead(in_features) for c in concepts
        })

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        presence_logits = []
        presence_probs = []
        severities = []

        for cid in self.concept_ids:
            p_logit, sev = self.heads[cid](z)
            presence_logits.append(p_logit)
            presence_probs.append(torch.sigmoid(p_logit))
            severities.append(sev)

        concept_logits = torch.stack(presence_logits, dim=1)      # (B, K)
        concept_probs = torch.stack(presence_probs, dim=1)        # (B, K)
        concept_severities = torch.stack(severities, dim=1)       # (B, K)

        # Concept vector combining calibrated presence probability with severity extent
        concept_vector = concept_probs * (0.6 + 0.4 * concept_severities)

        return {
            "concept_vector": concept_vector,
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            "concept_severities": concept_severities,
        }
