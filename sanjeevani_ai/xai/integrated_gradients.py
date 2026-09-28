"""
sanjeevani_ai.xai.integrated_gradients
--------------------------------------
Axiomatic Attribution for Deep Networks: Integrated Gradients (Sundararajan et al., ICML 2017).

Unlike heuristic saliency maps, Integrated Gradients satisfies two fundamental axioms:
1. Completeness: The attributions add up to the difference between the model's score for
   the input and its score for the baseline.
2. Implementation Invariance: Attributions are identical for functionally equivalent models.

Provides pixel-level resolution of tiny microaneurysms and subtle capillary shunts.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 30
) -> np.ndarray:
    """
    Computes Integrated Gradients attribution map for class_idx.
    Input:
        - model: nn.Module with forward returning {"logits": ...}
        - input_tensor: (1, 3, H, W)
        - class_idx: int
        - baseline: optional (1, 3, H, W), defaults to black image
        - steps: int (default 30 for low-latency laptop inference)
    Returns:
        - 2D attribution map (H, W) normalized to [0, 1].
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # 1. Generate linearly interpolated scaled inputs
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=input_tensor.device)
    scaled_inputs = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]

    # 2. Accumulate gradients across interpolated trajectory
    accumulated_grads = []
    for step_tensor in scaled_inputs:
        step_tensor = step_tensor.clone().detach().requires_grad_(True)
        out = model(step_tensor)
        score = out["logits"][:, class_idx].sum()
        grad = torch.autograd.grad(score, step_tensor, retain_graph=False, create_graph=False)[0]
        accumulated_grads.append(grad.detach())

    # 3. Riemann sum approximation
    avg_grads = torch.stack(accumulated_grads[:-1], dim=0).mean(dim=0)  # (1, 3, H, W)

    # 4. Multiply by input delta: (x - x_0) * avg_grad
    delta = input_tensor - baseline
    integrated_grad = delta * avg_grads  # (1, 3, H, W)

    # 5. Collapse across color channels (absolute magnitude)
    attribution_map = integrated_grad.abs().sum(dim=1).squeeze(0).cpu().numpy()

    # 6. Normalize to [0, 1]
    eps = 1e-8
    attr_min = attribution_map.min()
    attr_max = attribution_map.max()
    denom = attr_max - attr_min + eps
    norm_attribution = (attribution_map - attr_min) / denom

    return norm_attribution.astype(np.float32)
