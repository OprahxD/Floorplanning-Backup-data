import easyocr
import cv2
import numpy as np
import os
import re
import sys
import json

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "languages": ["en"],              # EasyOCR language list
    "min_confidence": 0.3,            # Discard detections below this score
    "gpu": False,                     # Set True if CUDA is available
    "numeric_only": True,             # If True, only keep detections that contain digits
}
# ---------------------------------------------------------------------------


def parse_numeric(text: str):
    """
    Extract the first numeric value (int or float) from an OCR string.
    Returns float or None if no number found.
    Examples: '3200' -> 3200.0 | '3,200' -> 3200.0 | '3.5m' -> 3.5
    """
    cleaned = text.replace(",", "").replace("'", "").strip()
    match = re.search(r"\d+(\.\d+)?", cleaned)
    return float(match.group()) if match else None


def _raw_ocr(img_rgb: np.ndarray, reader: easyocr.Reader) -> list[dict]:
    """
    Run EasyOCR on an RGB numpy array and return parsed detections.
    bbox is in terms of the supplied image's coordinate space.
    """
    detections = []
    for polygon, text, conf in reader.readtext(img_rgb):
        if conf < CONFIG["min_confidence"]:
            continue
        text = text.strip()
        digit_count = sum(c.isdigit() for c in text)
        if CONFIG["numeric_only"] and digit_count == 0:
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        detections.append({
            "text":          text,
            "numeric_value": parse_numeric(text),
            "digit_count":   digit_count,
            "confidence":    round(conf, 4),
            "bbox":          [x1, y1, x2, y2],
        })
    return detections


def _bbox_back(bbox: list[int], orig_h: int, orig_w: int, k: int) -> list[int]:
    """
    Map a bbox [x1,y1,x2,y2] from a np.rot90(img, k) rotated image back to
    the original image's coordinate space.

    k=1 (90° CCW):  new = ( y,       orig_w-1 - x )  → inv: orig = ( orig_w-1 - ny, nx )
    k=3 (90° CW):   new = ( orig_h-1 - y, x )         → inv: orig = ( ny, orig_h-1 - nx )
    """
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    if k == 1:   # image was rotated CCW → text was CW in original
        orig_corners = [(orig_w - 1 - cy, cx) for cx, cy in corners]
    elif k == 3: # image was rotated CW  → text was CCW in original
        orig_corners = [(cy, orig_h - 1 - cx) for cx, cy in corners]
    else:
        return bbox

    oxs = [c[0] for c in orig_corners]
    oys = [c[1] for c in orig_corners]
    return [int(min(oxs)), int(min(oys)), int(max(oxs)), int(max(oys))]


def run_ocr_on_image(image_path: str, reader: easyocr.Reader) -> list[dict]:
    """
    Run EasyOCR on a full floor plan image across three orientations:
      0°  — upright text
      90° CW  (k=3) — catches text rotated ACW in the original
      90° CCW (k=1) — catches text rotated CW  in the original

    Returns a list of dicts, one per detected text region:
    {
        "text":          raw OCR string,
        "numeric_value": parsed float (or None),
        "digit_count":   number of digit characters in text,
        "confidence":    float 0.0–1.0,
        "bbox":          [x1, y1, x2, y2]  in original image coordinates,
        "rotation":      0 | 90_CW | 90_CCW
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    passes = [
        (0,  "0"),       # upright
        (1,  "90_CCW"),  # image rotated CCW → catches CW text in original
        (3,  "90_CW"),   # image rotated CW  → catches CCW (ACW) text in original
    ]

    merged = []
    for k, rot_label in passes:
        rotated_img = np.rot90(img_rgb, k=k)
        for d in _raw_ocr(rotated_img, reader):
            d["bbox"]     = _bbox_back(d["bbox"], h, w, k) if k != 0 else d["bbox"]
            d["rotation"] = rot_label
            duplicate = any(iou(d["bbox"], m["bbox"]) > 0.4 and d["text"] == m["text"]
                            for m in merged)
            if not duplicate:
                merged.append(d)

    return merged


def run_ocr_on_crops_dir(crops_dir: str, reader: easyocr.Reader) -> list[dict]:
    """
    Run OCR on every image in a crops directory (output of csv2.py).
    Returns a list of dicts with an extra 'filename' key.
    """
    results = []
    supported = (".png", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(crops_dir) if f.lower().endswith(supported))

    for fname in files:
        path = os.path.join(crops_dir, fname)
        img = cv2.imread(path)
        if img is None:
            results.append({"filename": fname, "text": "", "numeric_value": None,
                             "digit_count": 0, "confidence": 0.0, "bbox": [], "rotation": -1})
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Try upright, 90° CCW, and 90° CW — pick the orientation with best numeric hit
        candidates = []
        for k, label in [(0, "0"), (1, "90_CCW"), (3, "90_CW")]:
            rotated = np.rot90(img_rgb, k=k)
            dets = _raw_ocr(rotated, reader)
            for d in dets:
                d["rotation"] = label
                candidates.append(d)

        numeric_hits = [c for c in candidates if c["numeric_value"] is not None]
        pool = numeric_hits if numeric_hits else candidates

        if pool:
            best = max(pool, key=lambda d: d["confidence"])
            best["filename"] = fname
            results.append(best)
        else:
            results.append({"filename": fname, "text": "", "numeric_value": None,
                             "digit_count": 0, "confidence": 0.0, "bbox": [], "rotation": -1})

    return results


def visualize_detections(image_path: str, detections: list[dict], save_path: str = None):
    """Draw bounding boxes and OCR text on the image."""
    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['text']} ({det['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Saved annotated image to: {save_path}")
    else:
        cv2.imshow("OCR Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# MAIN — demo entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    CROPS_DIR    = os.path.join(PROJECT_ROOT, "data/crops_number/output/crops")
    IMAGE_DIR    = os.path.join(PROJECT_ROOT, "Dataset/images")

    print("Initialising EasyOCR reader...")
    reader = easyocr.Reader(CONFIG["languages"], gpu=CONFIG["gpu"])

    # ── Mode 1: run on individual crops ──────────────────────────────────────
    if os.path.isdir(CROPS_DIR) and os.listdir(CROPS_DIR):
        print(f"\n[MODE] Running on crops in: {CROPS_DIR}")
        results = run_ocr_on_crops_dir(CROPS_DIR, reader)

        print(f"\n{'filename':<35} {'text':<12} {'numeric':>10} {'digits':>7} {'conf':>7} {'rot':>5}")
        print("-" * 82)
        for r in results:
            print(f"{r['filename']:<35} {r['text']:<12} "
                  f"{str(r['numeric_value']):>10} {r['digit_count']:>7} {r['confidence']:>7.3f} {r.get('rotation', '?'):>5}")

        # Save as JSON for downstream use
        out_path = os.path.join(PROJECT_ROOT, "data/crops_number/output/ocr_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved OCR results -> {out_path}")

    # ── Mode 2: run on full floor plan images ────────────────────────────────
    else:
        import glob
        img_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")) +
                           glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
        if not img_paths:
            print(f"No images found in {IMAGE_DIR}")
            sys.exit(1)

        img_path = img_paths[0]
        print(f"\n[MODE] Running on full image: {img_path}")
        detections = run_ocr_on_image(img_path, reader)

        print(f"\nFound {len(detections)} text region(s):\n")
        print(f"{'#':<4} {'text':<12} {'numeric':>10} {'digits':>7} {'conf':>7} {'rot':>5}  bbox")
        print("-" * 78)
        for i, d in enumerate(detections, 1):
            print(f"{i:<4} {d['text']:<12} {str(d['numeric_value']):>10} "
                  f"{d['digit_count']:>7} {d['confidence']:>7.3f} {d['rotation']:>5}  {d['bbox']}")

        save_path = os.path.join(PROJECT_ROOT, "data/crops_number/output/ocr_annotated.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_detections(img_path, detections, save_path=save_path)
