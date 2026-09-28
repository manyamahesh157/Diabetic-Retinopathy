"""
sanjeevani_ai.training.train
----------------------------
Modular Multi-Task Training Pipeline for Sanjeevani-AI.

Supports transfer learning on EfficientNet backbones with multi-task loss
(Focal Loss + Ordinal Regression + Multi-label Concept supervision).
"""

from typing import Dict, Any, Optional
import os
import torch
from torch.utils.data import DataLoader

from ..models.sanjeevani_model import SanjeevaniModel
from .losses import SanjeevaniMultiTaskLoss
from .dataset import RetinalDataset
from .evaluate import evaluate_screening_performance
from ..data.dataset_interface import create_demo_manifest_cohort, partition_manifests_by_patient


def train_sanjeevani(
    train_manifests: list,
    val_manifests: list,
    backbone_name: str = "efficientnet_b0",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes training loop with multi-task objective and validation tracking.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    model = SanjeevaniModel(backbone_name=backbone_name, pretrained=False).to(device)
    criterion = SanjeevaniMultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_ds = RetinalDataset(train_manifests, is_training=True)
    val_ds = RetinalDataset(val_manifests, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_qwk = -1.0
    best_checkpoint = os.path.join(checkpoint_dir, "sanjeevani_cbm_best.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            c_targets = batch["concept_target"].to(device)
            c_masks = batch["concept_mask"].to(device)

            optimizer.zero_grad()
            out = model(images)
            loss_dict = criterion(out, targets, c_targets, c_masks)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                out = model(images)
                probs = out["probs"].cpu().numpy()
                preds = probs.argmax(axis=1)

                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs)

        import numpy as np
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        metrics = evaluate_screening_performance(y_true, y_pred, y_probs)
        qwk = metrics["qwk"]

        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), best_checkpoint)

    return {
        "status": "completed",
        "best_qwk": best_qwk,
        "checkpoint_path": best_checkpoint,
    }


if __name__ == "__main__":
    print("Testing train script with demo cohort...")
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "sample_images")
    manifests = create_demo_manifest_cohort(samples_dir)
    train_m, val_m, _ = partition_manifests_by_patient(manifests, val_ratio=0.33, test_ratio=0.0)
    res = train_sanjeevani(train_m, val_m, epochs=1, batch_size=2)
    print("Training test result:", res)
