"""
Training script for the digit-count regressor.

Reads:   data/crops_number/output/number_labels.csv
         data/crops_number/output/crops/<filename>.png
Writes:  models/scale_pipeline/digit_regressor.pth  (best checkpoint by val accuracy)

Run:
    python models/scale_pipeline/train_digit_regressor.py
"""
import os
import sys

import cv2

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)

from utils.digit_regressor import (
    DigitCountCNN, preprocess_crop, MAX_DIGITS, INPUT_H, INPUT_W
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "csv_path":   os.path.join(PROJECT_ROOT, "data/crops_number/output/number_labels.csv"),
    "crops_dir":  os.path.join(PROJECT_ROOT, "data/crops_number/output/crops"),
    "out_model":  os.path.join(PROJECT_ROOT, "models/scale_pipeline/digit_regressor.pth"),
    "epochs":     100,
    "batch_size": 32,
    "lr":         1e-3,
    "val_split":  0.15,
    "seed":       42,
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class DigitCropDataset(Dataset):
    def __init__(self, csv_path: str, crops_dir: str):
        df = pd.read_csv(csv_path)
        df = df[(df["digit_count"] >= 1) & (df["digit_count"] <= MAX_DIGITS)]
        df = df.reset_index(drop=True)
        self.df        = df
        self.crops_dir = crops_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.crops_dir, row["filename"])
        crop_bgr = cv2.imread(img_path)

        if crop_bgr is None:
            # File missing — return a blank tensor so training can continue
            tensor = torch.zeros(1, INPUT_H, INPUT_W)
        else:
            tensor = preprocess_crop(crop_bgr).squeeze(0)   # 1 × H × W

        label = int(row["digit_count"]) - 1   # 0-indexed class
        return tensor, label


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    full_ds = DigitCropDataset(CONFIG["csv_path"], CONFIG["crops_dir"])
    print(f"Dataset: {len(full_ds)} valid crops  (digit counts 1–{MAX_DIGITS})")

    if len(full_ds) == 0:
        print("ERROR: No samples found. Run data/crops_number/csv2.py first.")
        sys.exit(1)

    n_val   = max(1, int(len(full_ds) * CONFIG["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model / optimizer ─────────────────────────────────────────────────────
    model     = DigitCountCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"])

    best_val_acc = 0.0

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, CONFIG["epochs"] + 1):
        # Train
        model.train()
        t_loss, t_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(imgs)
            t_correct += (logits.argmax(1) == labels).sum().item()
        scheduler.step()

        # Validate
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                v_loss    += criterion(logits, labels).item() * len(imgs)
                v_correct += (logits.argmax(1) == labels).sum().item()

        t_acc = t_correct / n_train
        v_acc = v_correct / n_val
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']}  "
              f"train loss={t_loss/n_train:.4f}  acc={t_acc:.3f}  |  "
              f"val loss={v_loss/n_val:.4f}  acc={v_acc:.3f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CONFIG["out_model"])
            print(f"             -> saved (val_acc={v_acc:.3f})")

    print(f"\nDone. Best val_acc = {best_val_acc:.3f}")
    print(f"Model -> {CONFIG['out_model']}")


if __name__ == "__main__":
    train()
