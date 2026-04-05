import numpy as np
import cv2
import os
import sys
import json
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "max_match_distance":   150,   # Max pixels from text bbox center to line segment
    "orientation_filter":   True,  # Match text rotation to line orientation
    "axis_tolerance":       10,    # Same tolerance used in pair_endpoints
    "use_digit_regressor":  True,  # Pre-filter OCR results with digit-count regressor
    "regressor_model_path": None,  # Set to path string to override default location
}
# ---------------------------------------------------------------------------


# ── Geometry helpers ────────────────────────────────────────────────────────

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pt_to_segment_dist(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px,py) to finite segment (x1,y1)→(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _is_vertical(p1, p2, tol):
    return abs(p1[0] - p2[0]) <= tol


def _orientation_ok(p1, p2, text_rotation, tol):
    """
    Upright text (rotation '0')      → should belong to a horizontal line.
    Rotated text ('90_CW'/'90_CCW')  → should belong to a vertical line.
    If orientation_filter is off, always True.
    """
    if not CONFIG["orientation_filter"]:
        return True
    line_is_vertical = _is_vertical(p1, p2, tol)
    text_is_rotated  = text_rotation in ("90_CW", "90_CCW")
    return line_is_vertical == text_is_rotated


# ── Core matching ────────────────────────────────────────────────────────────

def match_lines_to_text(pairs, ocr_results):
    """
    Greedily match each dimension line segment to its nearest OCR text label.

    Each text can only be claimed by one line (greedy, sorted by distance).

    Args:
        pairs:       list of ((x1,y1),(x2,y2)) — output of pair_endpoints()
        ocr_results: list of dicts — output of run_ocr_on_image() / run_ocr_on_crops_dir()
                     each dict must have: bbox, numeric_value, rotation, confidence

    Returns:
        matches        — list of match dicts (see below)
        unmatched_lines — lines that found no text within max_match_distance
        unmatched_texts — OCR results not claimed by any line
    """
    valid_texts = [
        r for r in ocr_results
        if r.get("numeric_value") is not None and r.get("bbox")
    ]

    tol      = CONFIG["axis_tolerance"]
    max_dist = CONFIG["max_match_distance"]

    # Build all (distance, line_idx, text_idx) candidates first, then greedily assign
    candidates = []
    for li, (p1, p2) in enumerate(pairs):
        for ti, text in enumerate(valid_texts):
            if not _orientation_ok(p1, p2, text.get("rotation", "0"), tol):
                continue
            cx, cy = _bbox_center(text["bbox"])
            dist = _pt_to_segment_dist(cx, cy, p1[0], p1[1], p2[0], p2[1])
            if dist <= max_dist:
                candidates.append((dist, li, ti))

    candidates.sort(key=lambda c: c[0])   # closest pairs first

    used_lines = set()
    used_texts = set()
    matches    = []

    for dist, li, ti in candidates:
        if li in used_lines or ti in used_texts:
            continue
        p1, p2 = pairs[li]
        text   = valid_texts[ti]
        px_len = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

        matches.append({
            "line":           (p1, p2),
            "px_length":      round(px_len, 2),
            "text_value":     text["numeric_value"],
            "text_raw":       text.get("text", ""),
            "text_bbox":      text["bbox"],
            "text_rotation":  text.get("rotation", "0"),
            "distance_px":    round(dist, 2),
            "ocr_confidence": text.get("confidence", 0.0),
        })
        used_lines.add(li)
        used_texts.add(ti)

    unmatched_lines = [
        {"line": pairs[i], "px_length": round(np.hypot(pairs[i][1][0]-pairs[i][0][0],
                                                        pairs[i][1][1]-pairs[i][0][1]), 2)}
        for i in range(len(pairs)) if i not in used_lines
    ]
    unmatched_texts = [valid_texts[i] for i in range(len(valid_texts)) if i not in used_texts]

    return matches, unmatched_lines, unmatched_texts


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_matches(image_path, matches, unmatched_lines=None, save_path=None):
    """
    Green lines  = matched (with text value label).
    Red dashed   = unmatched lines.
    Green rect   = text bounding box used for the match.
    """
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(img)

    for m in matches:
        (x1, y1), (x2, y2) = m["line"]
        ax.plot([x1, x2], [y1, y2], color="lime", linewidth=2.5, zorder=3)
        ax.scatter([x1, x2], [y1, y2], color="cyan", s=25, zorder=4)

        # Value label at line midpoint
        ax.text((x1+x2)/2, (y1+y2)/2 - 10, str(m["text_value"]),
                color="yellow", fontsize=7, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6), zorder=5)

        # OCR text bbox
        bx1, by1, bx2, by2 = m["text_bbox"]
        ax.add_patch(mpatches.Rectangle(
            (bx1, by1), bx2-bx1, by2-by1,
            linewidth=1.2, edgecolor="lime", facecolor="none", zorder=3))

    for ul in (unmatched_lines or []):
        (x1, y1), (x2, y2) = ul["line"]
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.5,
                linestyle="--", zorder=2)

    n_matched = len(matches)
    n_total   = n_matched + len(unmatched_lines or [])
    ax.set_title(f"Scale Matching  |  {n_matched}/{n_total} lines matched", fontsize=13)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualisation -> {save_path}")
    else:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, PROJECT_ROOT)

    from utils.endpoint_fcn      import EndpointFCN
    from utils.extract_endpoints import extract_endpoints_from_heatmap
    from utils.pair_endpoints    import pair_endpoints
    from utils.ocr_reader        import run_ocr_on_image
    from utils.digit_regressor   import load_regressor, filter_ocr_by_regressor
    import easyocr
    from torchvision import transforms
    from PIL import Image

    MODEL_PATH      = os.path.join(PROJECT_ROOT, "models/scale_pipeline/endpoint_fcn_final.pth")
    REGRESSOR_PATH  = (CONFIG["regressor_model_path"] or
                       os.path.join(PROJECT_ROOT, "models/scale_pipeline/digit_regressor.pth"))
    IMAGE_DIR  = os.path.join(PROJECT_ROOT, "Dataset/images")
    OUT_DIR    = os.path.join(PROJECT_ROOT, "data/crops_number/output")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load FCN ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EndpointFCN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"FCN loaded on {device}")

    # ── Pick first image ──────────────────────────────────────────────────────
    img_paths = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.png")) +
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    )
    if not img_paths:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    img_path = img_paths[0]
    print(f"Image: {img_path}")

    # ── FCN inference → endpoints → pairs ────────────────────────────────────
    pil_img = Image.open(img_path).convert("RGB")
    tensor  = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = model(tensor)

    endpoints = extract_endpoints_from_heatmap(heatmap, threshold=0.3, kernel_size=5)
    pairs, unpaired = pair_endpoints(endpoints, axis_tolerance=10,
                                     min_length=20, max_length=480)
    print(f"Pairs: {len(pairs)}  |  Unpaired endpoints: {len(unpaired)}")

    # ── OCR ───────────────────────────────────────────────────────────────────
    print("Initialising EasyOCR...")
    reader      = easyocr.Reader(["en"], gpu=False)
    ocr_results = run_ocr_on_image(img_path, reader)
    print(f"OCR detections: {len(ocr_results)}")

    # ── Digit-regressor double-check ──────────────────────────────────────────
    if CONFIG["use_digit_regressor"] and os.path.exists(REGRESSOR_PATH):
        print(f"Loading digit-count regressor from {REGRESSOR_PATH} ...")
        regressor   = load_regressor(REGRESSOR_PATH, device)
        image_bgr   = cv2.imread(img_path)
        ocr_results, rejected = filter_ocr_by_regressor(ocr_results, image_bgr,
                                                         regressor, device)
        print(f"Regressor kept {len(ocr_results)} / "
              f"{len(ocr_results) + len(rejected)} OCR detections")
        if rejected:
            print("  Rejected:")
            for r in rejected:
                print(f"    '{r.get('text','')}' — {r.get('reject_reason','')}")
    elif CONFIG["use_digit_regressor"]:
        print(f"[WARN] Digit regressor enabled but model not found at {REGRESSOR_PATH}.")
        print("       Run models/scale_pipeline/train_digit_regressor.py first.")
        print("       Proceeding without Double-Check filter.")

    # ── Scale matching ────────────────────────────────────────────────────────
    matches, unmatched_lines, unmatched_texts = match_lines_to_text(pairs, ocr_results)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Scale Matching Results")
    print(f"{'='*65}")
    print(f"Matched {len(matches)} / {len(pairs)} lines to text labels\n")

    print(f"  {'#':<4} {'px_len':>8} {'value':>10} {'dist_px':>9} {'rot':<10} {'conf':>6}  line")
    print(f"  {'-'*70}")
    for i, m in enumerate(matches, 1):
        (x1,y1),(x2,y2) = m["line"]
        print(f"  {i:<4} {m['px_length']:>8.1f} {m['text_value']:>10} "
              f"{m['distance_px']:>9.1f} {m['text_rotation']:<10} "
              f"{m['ocr_confidence']:>6.3f}  ({x1},{y1})→({x2},{y2})")

    if unmatched_lines:
        print(f"\n  {len(unmatched_lines)} unmatched line(s) — no text within "
              f"{CONFIG['max_match_distance']}px:")
        for ul in unmatched_lines:
            (x1,y1),(x2,y2) = ul["line"]
            print(f"    ({x1},{y1})→({x2},{y2})  len={ul['px_length']}px")

    if unmatched_texts:
        print(f"\n  {len(unmatched_texts)} unmatched text detection(s):")
        for ut in unmatched_texts:
            print(f"    '{ut.get('text','')}' = {ut.get('numeric_value')}  "
                  f"bbox={ut.get('bbox')}  rot={ut.get('rotation')}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "scale_matches.json")
    serialisable = [
        {**m, "line": [list(m["line"][0]), list(m["line"][1])]}
        for m in matches
    ]
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nSaved matches -> {out_json}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    vis_path = os.path.join(OUT_DIR, "scale_matches.png")
    visualize_matches(img_path, matches, unmatched_lines, save_path=vis_path)
