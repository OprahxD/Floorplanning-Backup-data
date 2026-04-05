"""
Evaluation script for the digit-count regressor.

Recreates the same train/val split used in training (same seed) so
the validation set is guaranteed to be unseen data.

Outputs:
  - Per-class accuracy breakdown
  - Confusion matrix saved to models/scale_pipeline/digit_regressor_confusion.png
  - Per-sample listing of every wrong prediction

Run:
    python models/scale_pipeline/eval_digit_regressor.py
"""
import os
import sys

import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)

from utils.digit_regressor import (
    DigitCountCNN, preprocess_crop, MAX_DIGITS, INPUT_H, INPUT_W, load_regressor
)
from models.scale_pipeline.train_digit_regressor import DigitCropDataset

# ── Config — must match train_digit_regressor.py ──────────────────────────────
CONFIG = {
    "csv_path":    os.path.join(PROJECT_ROOT, "data/crops_number/output/number_labels.csv"),
    "crops_dir":   os.path.join(PROJECT_ROOT, "data/crops_number/output/crops"),
    "model_path":  os.path.join(PROJECT_ROOT, "models/scale_pipeline/digit_regressor.pth"),
    "val_split":   0.15,
    "seed":        42,
    "batch_size":  32,
}
OUT_DIR = os.path.join(os.path.dirname(__file__))


def rebuild_val_set():
    """Recreate the identical val split used during training."""
    full_ds = DigitCropDataset(CONFIG["csv_path"], CONFIG["crops_dir"])
    n_val   = max(1, int(len(full_ds) * CONFIG["val_split"]))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    return val_ds, full_ds.df


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CONFIG["model_path"]):
        print(f"ERROR: Model not found at {CONFIG['model_path']}")
        print("Run train_digit_regressor.py first.")
        sys.exit(1)

    model = load_regressor(CONFIG["model_path"], device)
    print(f"Model loaded from {CONFIG['model_path']}")

    val_ds, full_df = rebuild_val_set()
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=2)
    print(f"Validation set: {len(val_ds)} samples\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_labels: list[int] = []
    all_preds:  list[int] = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_labels = np.array(all_labels)   # 0-indexed classes
    all_preds  = np.array(all_preds)

    # Convert back to digit counts (1-indexed) for display
    true_counts = all_labels + 1
    pred_counts = all_preds  + 1

    # ── Summary ───────────────────────────────────────────────────────────────
    overall_acc = (all_labels == all_preds).mean()
    print("=" * 55)
    print(f"  Overall accuracy:  {overall_acc:.3f}  "
          f"({(all_labels == all_preds).sum()} / {len(all_labels)} correct)")
    print("=" * 55)

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n  {'Digit count':<14} {'N samples':>10} {'Correct':>9} {'Acc':>7}")
    print(f"  {'-'*44}")
    classes = sorted(set(true_counts.tolist()))
    for dc in classes:
        mask    = true_counts == dc
        n       = mask.sum()
        correct = (pred_counts[mask] == dc).sum()
        acc     = correct / n if n > 0 else 0.0
        bar     = "#" * int(acc * 20)
        print(f"  {dc} digit(s)      {n:>10}  {correct:>9}  {acc:>6.1%}  {bar}")

    # ── Wrong predictions ─────────────────────────────────────────────────────
    wrong_mask = all_labels != all_preds
    n_wrong    = wrong_mask.sum()
    print(f"\n  Wrong predictions: {n_wrong}")

    if n_wrong > 0:
        # Reconstruct which val-set rows correspond to wrong predictions
        val_indices = list(val_ds.indices)                     # original full_df row indices
        wrong_positions = np.where(wrong_mask)[0]

        print(f"\n  {'#':<5} {'File':<40} {'True':>6} {'Pred':>6}")
        print(f"  {'-'*60}")
        for pos in wrong_positions:
            orig_idx  = val_indices[pos]
            row       = full_df.iloc[orig_idx]
            true_dc   = int(row["digit_count"])
            pred_dc   = int(all_preds[pos]) + 1
            fname     = str(row["filename"])
            print(f"  {pos:<5} {fname:<40} {true_dc:>6} {pred_dc:>6}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    digit_range  = list(range(1, MAX_DIGITS + 1))
    n_classes    = len(digit_range)
    conf_matrix  = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(true_counts, pred_counts):
        ti = t - 1   # row = true class
        pi = p - 1   # col = predicted class
        if 0 <= ti < n_classes and 0 <= pi < n_classes:
            conf_matrix[ti, pi] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap="Blues")
    plt.colorbar(im, ax=ax, label="count")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels([f"{d}" for d in digit_range])
    ax.set_yticklabels([f"{d}" for d in digit_range])
    ax.set_xlabel("Predicted digit count")
    ax.set_ylabel("True digit count")
    ax.set_title(f"Digit Regressor — Confusion Matrix\n"
                 f"Val accuracy: {overall_acc:.3f}  ({len(val_ds)} samples)")

    # Cell annotations
    for i in range(n_classes):
        for j in range(n_classes):
            val = conf_matrix[i, j]
            if val > 0:
                color = "white" if val > conf_matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "digit_regressor_confusion.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved -> {out_png}")
    plt.close()


if __name__ == "__main__":
    evaluate()
