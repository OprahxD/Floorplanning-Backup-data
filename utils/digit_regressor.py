"""
Digit-count regressor: a small CNN that predicts how many digit characters
appear in a cropped dimension-text image.

Used as a "Double Check" against EasyOCR's own digit count:
  - OCR says "3200"  → digit_count = 4
  - Regressor sees the crop and predicts 4  → ✓ keep
  - Regressor predicts 2                    → ✗ discard (likely OCR error)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DIGITS  = 8    # Support 1–8 digit numbers (floor-plan dimensions)
INPUT_H     = 32   # All crops resized to this height
INPUT_W     = 128  # Wide enough to fit "30000" etc.
IN_CHANNELS = 1    # Grayscale


# ── Model ─────────────────────────────────────────────────────────────────────

class DigitCountCNN(nn.Module):
    """
    3 conv blocks + global average pool + FC head.
    Output: logits over [1 .. MAX_DIGITS] digit-count classes.
    Class index 0 → 1 digit, index 1 → 2 digits, …
    """
    def __init__(self, n_classes: int = MAX_DIGITS):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 → 32   |  32 × 128
            nn.Conv2d(IN_CHANNELS, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                         # 16 × 64
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                         # 8 × 32
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),            # 1 × 1  (global avg pool)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR numpy crop  →  (1, 1, INPUT_H, INPUT_W) float32 tensor, normalised [0, 1].
    """
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    tensor  = torch.from_numpy(resized).float() / 255.0   # H × W
    return tensor.unsqueeze(0).unsqueeze(0)               # 1 × 1 × H × W


# ── Inference ─────────────────────────────────────────────────────────────────

def load_regressor(model_path: str, device: torch.device) -> DigitCountCNN:
    """Load a saved DigitCountCNN checkpoint."""
    model = DigitCountCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def predict_digit_count(model: DigitCountCNN,
                        crop_bgr: np.ndarray,
                        device: torch.device) -> int:
    """
    Returns predicted digit count as a 1-indexed integer.
    e.g. argmax = 3  →  4 digits.
    """
    tensor = preprocess_crop(crop_bgr).to(device)
    with torch.no_grad():
        logits = model(tensor)
    return int(torch.argmax(logits, dim=1).item()) + 1   # class 0 = 1 digit


# ── Double-check filter ───────────────────────────────────────────────────────

def filter_ocr_by_regressor(
    ocr_results: list[dict],
    image_bgr: np.ndarray,
    model: DigitCountCNN,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    """
    For each OCR result that carries a bbox, crop from image_bgr, run the
    regressor, and keep only the results where the predicted digit count
    matches the OCR digit count.

    Args:
        ocr_results : list of dicts from ocr_reader (must have 'bbox' and 'digit_count')
        image_bgr   : the full floor-plan image as a BGR numpy array
        model       : a loaded DigitCountCNN in eval mode
        device      : torch device

    Returns:
        verified  — results where regressor agrees with OCR digit_count
        rejected  — results where they disagree (or the crop was invalid)
    """
    h_img, w_img = image_bgr.shape[:2]
    verified: list[dict] = []
    rejected: list[dict] = []

    for r in ocr_results:
        bbox       = r.get("bbox")
        ocr_digits = r.get("digit_count", 0)

        # Can't verify without a bbox or if OCR found no digits
        if not bbox or ocr_digits == 0:
            rejected.append({**r, "regressor_predicted": None,
                             "reject_reason": "no_bbox_or_no_digits"})
            continue

        x1, y1, x2, y2 = (max(0, int(bbox[0])), max(0, int(bbox[1])),
                           min(w_img, int(bbox[2])), min(h_img, int(bbox[3])))
        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            rejected.append({**r, "regressor_predicted": None,
                             "reject_reason": "empty_crop"})
            continue

        predicted = predict_digit_count(model, crop, device)

        if predicted == ocr_digits:
            verified.append({**r, "regressor_predicted": predicted})
        else:
            rejected.append({**r, "regressor_predicted": predicted,
                             "reject_reason": f"ocr={ocr_digits}_reg={predicted}"})

    return verified, rejected
