"""
sanjeevani_ai.counterfactual.concept_counterfactual
---------------------------------------------------
Concept-Space Counterfactual Optimization (Wachter et al., 2017 / Concept Adaptation).

Answers the key clinical question:
"What minimal changes in retinal lesion evidence would cause the model
to assign an adjacent severity grade?"

Formulation:
   min_{delta c}  ||delta c||_2^2 + lambda * ||delta c||_1 + beta * CE(g_phi(clip(c + delta c, 0, 1)), y_target)

Because optimization takes place directly in the human-interpretable concept space c in [0, 1]^K,
the resulting counterfactual is immediately meaningful to clinicians (e.g. -25% hemorrhages).

IMPORTANT:
This is a mathematical model counterfactual demonstrating sensitivity of the decision boundary;
it is NOT a biological claim about patient progression or disease reversal.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from ..models.concept_bottleneck import CLINICAL_CONCEPTS
from ..models.severity_head import ICDR_GRADES


def optimize_concept_counterfactual(
    severity_head: torch.nn.Module,
    current_concept_vector: torch.Tensor,
    current_grade: int,
    target_grade: int,
    max_steps: int = 80,
    lr: float = 0.05,
    l1_weight: float = 0.02,
    beta: float = 4.0,
    concept_specs: List[Dict[str, Any]] = CLINICAL_CONCEPTS,
) -> Dict[str, Any]:
    """
    Finds minimal perturbation delta_c to transition severity head prediction
    from current_grade to target_grade.
    """
    device = current_concept_vector.device
    c_orig = current_concept_vector.detach().clone()
    if c_orig.ndim == 1:
        c_orig = c_orig.unsqueeze(0)

    # Initialize perturbation parameter
    delta = torch.zeros_like(c_orig, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)

    target_tensor = torch.tensor([target_grade], device=device, dtype=torch.long)
    converged = False
    best_delta = delta.detach().clone()
    min_loss = float("inf")

    for step in range(max_steps):
        optimizer.zero_grad()
        # Constrain c_cf in [0.0, 1.0]
        c_perturbed = torch.clamp(c_orig + delta, 0.0, 1.0)

        logits, _ = severity_head(c_perturbed)
        loss_ce = F.cross_entropy(logits, target_tensor)
        loss_l2 = torch.norm(delta, p=2)
        loss_l1 = torch.norm(delta, p=1)

        total_loss = beta * loss_ce + loss_l2 + l1_weight * loss_l1
        total_loss.backward()
        optimizer.step()

        pred_grade = int(torch.argmax(logits, dim=-1).item())
        if pred_grade == target_grade:
            converged = True
            if total_loss.item() < min_loss:
                min_loss = total_loss.item()
                best_delta = delta.detach().clone()

    final_delta = best_delta if converged else delta.detach()
    final_c = torch.clamp(c_orig + final_delta, 0.0, 1.0).squeeze(0).cpu().numpy()
    delta_np = (final_c - c_orig.squeeze(0).cpu().numpy())

    # Build structured explanation of concept changes
    changes = []
    for idx, spec in enumerate(concept_specs):
        orig_val = float(c_orig[0, idx].item())
        new_val = float(final_c[idx])
        change_val = float(delta_np[idx])
        abs_change = abs(change_val)

        if abs_change >= 0.05:  # Only report meaningful shifts
            direction = "increase" if change_val > 0 else "decrease"
            changes.append({
                "concept_id": spec["id"],
                "name": spec["name"],
                "original_value": round(orig_val, 3),
                "counterfactual_value": round(new_val, 3),
                "delta": round(change_val, 3),
                "abs_delta": round(abs_change, 3),
                "direction": direction,
                "summary": f"{spec['name']}: {direction} by {abs_change * 100:.1f}%",
            })

    # Sort by magnitude of required change
    changes.sort(key=lambda x: x["abs_delta"], reverse=True)

    summary_sentence = (
        f"To shift prediction from {ICDR_GRADES[current_grade]} to {ICDR_GRADES[target_grade]}, "
        f"the model is most sensitive to changes in: "
        + (", ".join([c["name"] for c in changes[:3]]) if changes else "minor distributed concept shifts")
        + "."
    )

    return {
        "success": converged,
        "current_grade": current_grade,
        "current_grade_name": ICDR_GRADES[current_grade],
        "target_grade": target_grade,
        "target_grade_name": ICDR_GRADES[target_grade],
        "counterfactual_concept_vector": final_c.tolist(),
        "concept_changes": changes,
        "primary_sensitive_concepts": [c["name"] for c in changes[:3]],
        "summary": summary_sentence,
        "disclaimer": "MODEL COUNTERFACTUAL: Reflects mathematical sensitivity of the decision boundary; not a biological claim.",
    }


def compute_borderline_counterfactuals(
    severity_head: torch.nn.Module,
    concept_vector: torch.Tensor,
    predicted_grade: int,
    concept_specs: List[Dict[str, Any]] = CLINICAL_CONCEPTS,
) -> Dict[str, Any]:
    """
    Computes both downward (milder) and upward (more severe) adjacent counterfactuals.
    """
    downward_cf = None
    upward_cf = None

    if predicted_grade > 0:
        downward_cf = optimize_concept_counterfactual(
            severity_head, concept_vector,
            current_grade=predicted_grade,
            target_grade=predicted_grade - 1,
            concept_specs=concept_specs
        )

    if predicted_grade < 4:
        upward_cf = optimize_concept_counterfactual(
            severity_head, concept_vector,
            current_grade=predicted_grade,
            target_grade=predicted_grade + 1,
            concept_specs=concept_specs
        )

    return {
        "predicted_grade": predicted_grade,
        "predicted_grade_name": ICDR_GRADES[predicted_grade],
        "downward_transition": downward_cf,
        "upward_transition": upward_cf,
    }
