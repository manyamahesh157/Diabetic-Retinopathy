"""
sanjeevani_ai.training.dataset
------------------------------
PyTorch Dataset Implementation for Retinal Fundus Screening.
"""

from typing import List, Optional
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from ..data.dataset_interface import DatasetManifest
from ..preprocessing.pipeline import PreprocessingPipeline
from ..models.concept_bottleneck import CLINICAL_CONCEPTS


class RetinalDataset(Dataset):
    def __init__(
        self,
        manifests: List[DatasetManifest],
        img_size: int = 384,
        normalization_method: str = "ben_graham",
        is_training: bool = False
    ):
        self.manifests = manifests
        self.pipeline = PreprocessingPipeline(target_size=img_size, normalization_method=normalization_method)
        self.is_training = is_training
        self.concept_keys = [c["id"] for c in CLINICAL_CONCEPTS]

    def __len__(self) -> int:
        return len(self.manifests)

    def __getitem__(self, idx: int):
        m = self.manifests[idx]

        # Read image
        if os.path.exists(m.image_path):
            img_bgr = cv2.imread(m.image_path)
        else:
            # Fallback black image if missing
            img_bgr = np.zeros((384, 384, 3), dtype=np.uint8)

        # Preprocessing (bypass quality gate during dataset training batching, but crop & normalize)
        res = self.pipeline.process(img_bgr, bypass_quality_gate=True)
        tensor = res["tensor"].squeeze(0)  # (3, H, W)

        # DR grade target
        grade_target = torch.tensor(m.grade, dtype=torch.long)

        # Concept targets & masks
        c_targets = []
        c_masks = []
        for ck in self.concept_keys:
            if ck in m.concepts:
                c_targets.append(float(m.concepts[ck]))
                c_masks.append(1.0)
            else:
                c_targets.append(0.0)
                c_masks.append(0.0)

        concept_target_tensor = torch.tensor(c_targets, dtype=torch.float32)
        concept_mask_tensor = torch.tensor(c_masks, dtype=torch.float32)

        return {
            "image": tensor,
            "target": grade_target,
            "concept_target": concept_target_tensor,
            "concept_mask": concept_mask_tensor,
            "patient_id": m.patient_id,
            "eye_side": m.eye_side,
            "dataset_source": m.dataset_source,
        }
