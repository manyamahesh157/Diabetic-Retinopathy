"""
sanjeevani_ai.xai.gradcam
-------------------------
Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks
(Chattopadhay et al., WACV 2018).

Specifically addresses standard Grad-CAM's inability to isolate multiple instances of small
lesions (such as scattered microaneurysms and dot hemorrhages across the retinal field).
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []

        self._handles.append(target_layer.register_forward_hook(self._save_activation))
        self._handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Computes Grad-CAM++ saliency map for class_idx.
        Returns: 2D numpy array (H, W) in [0.0, 1.0].
        """
        self.model.zero_grad()
        out = self.model(input_tensor)
        logits = out["logits"]
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't trigger
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)

        grads = self.gradients                       # (B, C, H, W)
        acts = self.activations                       # (B, C, H, W)

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        eps = 1e-8

        # Spatial sum of acts * grads^3
        sum_acts_grads3 = (acts * grads_power_3).sum(dim=[2, 3], keepdim=True)
        alpha_denom = 2.0 * grads_power_2 + sum_acts_grads3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
        alphas = grads_power_2 / alpha_denom

        weights = (alphas * F.relu(grads)).sum(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        denom = cam_np.max() - cam_np.min() + eps
        cam_norm = (cam_np - cam_np.min()) / denom
        return cam_norm.astype(np.float32)


def overlay_heatmap(
    img_rgb_float: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays 2D heatmap on RGB image in [0, 1].
    Returns uint8 RGB image [0, 255].
    """
    h, w = img_rgb_float.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    base = img_rgb_float.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * heatmap_rgb
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
