"""
sanjeevani_ai.preprocessing.pipeline
------------------------------------
End-to-end Preprocessing Pipeline:
Raw BGR Image / Uploaded Bytes -> Quality Gate -> FOV Crop -> Normalization -> Model Tensor.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch

from .quality import check_image_gradability, QualityReport
from .crop import crop_to_retinal_fov
from .normalization import apply_ben_graham_normalization, apply_green_channel_clahe, normalize_to_tensor


class PreprocessingPipeline:
    def __init__(
        self,
        target_size: int = 384,
        normalization_method: str = "ben_graham",
        quality_config: Optional[Dict[str, Any]] = None
    ):
        self.target_size = target_size
        self.normalization_method = normalization_method
        self.quality_config = quality_config or {}

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decodes uploaded byte stream to OpenCV BGR numpy array."""
        file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img_bgr

    def process(
        self,
        img_bgr: np.ndarray,
        bypass_quality_gate: bool = False
    ) -> Dict[str, Any]:
        """
        Executes full preprocessing:
        Returns:
            - success: bool
            - quality_report: QualityReport
            - original_rgb: np.ndarray (uint8)
            - cropped_rgb: np.ndarray (uint8)
            - normalized_rgb: np.ndarray (uint8)
            - normalized_float: np.ndarray (float32 in [0, 1])
            - tensor: torch.Tensor (1, 3, H, W)
            - error: Optional[str]
        """
        if img_bgr is None or img_bgr.size == 0:
            return {
                "success": False,
                "error": "Failed to decode image. Please provide a valid JPG/PNG fundus photo.",
                "quality_report": None,
            }

        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Quality Gate Check
        quality_report = check_image_gradability(
            img_bgr,
            min_laplacian_var=self.quality_config.get("min_laplacian_var", 45.0),
            min_mean_brightness=self.quality_config.get("min_mean_brightness", 20.0),
            max_mean_brightness=self.quality_config.get("max_mean_brightness", 215.0),
            max_glare_percentage=self.quality_config.get("max_glare_percentage", 8.0),
            min_fov_fraction=self.quality_config.get("min_fov_fraction", 0.18),
            min_rb_ratio=self.quality_config.get("min_rb_ratio", 1.65),
        )

        if not quality_report.is_gradable and not bypass_quality_gate:
            return {
                "success": False,
                "error": "Image rejected by Clinical Quality Gate.",
                "quality_report": quality_report,
                "original_rgb": original_rgb,
                "cropped_rgb": None,
                "normalized_rgb": None,
                "normalized_float": None,
                "tensor": None,
            }

        # 2. Retinal FOV Auto-Crop
        cropped_bgr, mask, bbox = crop_to_retinal_fov(img_bgr)
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

        # 3. Resize to target dimension
        resized_rgb = cv2.resize(
            cropped_rgb,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA
        )

        # 4. Illumination / Color Normalization
        if self.normalization_method == "ben_graham":
            normalized_rgb = apply_ben_graham_normalization(resized_rgb, sigma=10)
        elif self.normalization_method == "clahe":
            normalized_rgb = apply_green_channel_clahe(resized_rgb)
        else:
            normalized_rgb = resized_rgb.copy()

        # 5. Float conversion [0, 1] and PyTorch tensor
        normalized_float = normalized_rgb.astype(np.float32) / 255.0
        tensor = normalize_to_tensor(normalized_float).unsqueeze(0)  # (1, 3, H, W)

        return {
            "success": True,
            "error": None,
            "quality_report": quality_report,
            "original_rgb": original_rgb,
            "cropped_rgb": cropped_rgb,
            "normalized_rgb": normalized_rgb,
            "normalized_float": normalized_float,
            "tensor": tensor,
            "crop_bbox": bbox,
        }
