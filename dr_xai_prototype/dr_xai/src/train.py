"""
train.py
--------
Reference training script. Expects a CSV with columns [image_path, label]
(this matches the APTOS-2019 Blindness Detection dataset format on Kaggle,
which is the standard public dataset for this problem — ~3,662 labeled
fundus images across the 5 ICDR grades). Point --data-csv at your own
data of the same format.

Algorithms used (see model.py / preprocessing.py for details):
  - Transfer learning (EfficientNet-B3, ImageNet-pretrained)
  - Albumentations augmentation (rotation, flips, brightness/contrast,
    elastic/grid distortion to simulate camera variation)
  - Weighted random sampling + Focal Loss + ordinal penalty for the heavy
    class imbalance always present in DR datasets (Grade-0 is ~50%+ of data)
  - Metric: Quadratic Weighted Kappa (QWK) — the official metric used by
    the APTOS/EyePACS Kaggle competitions, because plain accuracy is
    misleading on an ordinal, imbalanced 5-class problem.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score
import albumentations as A
import cv2

from . import config
from .model import DRGradingModel, combined_loss
from .preprocessing import preprocess_fundus_image, to_model_tensor


TRAIN_AUG = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, p=0.2),
])


class DRDataset(Dataset):
    def __init__(self, csv_path: str, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(row["image_path"])
        img = preprocess_fundus_image(img_bgr, config.IMG_SIZE)
        if self.train:
            img_uint8 = (img * 255).astype(np.uint8)
            img_uint8 = TRAIN_AUG(image=img_uint8)["image"]
            img = img_uint8.astype(np.float32) / 255.0
        tensor = to_model_tensor(img)
        label = int(row["label"])
        return tensor, label


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def evaluate(model, loader, device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            targets.extend(y.numpy().tolist())
    return cohen_kappa_score(targets, preds, weights="quadratic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    device = config.DEVICE
    train_ds = DRDataset(args.data_csv, train=True)
    val_ds = DRDataset(args.val_csv, train=False)

    sampler = make_weighted_sampler(train_ds.df["label"].values)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = DRGradingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = combined_loss(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        qwk = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={running_loss/len(train_ds):.4f} | val_QWK={qwk:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            import os
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (QWK={qwk:.4f})")

    print(f"Training complete. Best validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
