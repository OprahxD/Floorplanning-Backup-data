"""
run_pipeline.py — End-to-end dimension extraction for a single floor plan image.

Usage:
    python run_pipeline.py <image_path> [--out-dir <dir>] [--no-regressor] [--no-vis]

Steps:
    1. FCN inference           → endpoint heatmap
    2. NMS peak extraction     → (x, y) endpoints
    3. Line pairing            → axis-aligned dimension segments
    4. EasyOCR                 → text detections with bounding boxes
    5. Digit-count regressor   → discard OCR reads where digit count disagrees
    6. Scale matching          → link each line to its nearest text label
    7. K-Means scale voting    → consensus px-per-unit ratio
    8. Final output            → real-world dimensions (JSON + optional visualisation)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Resolve project root regardless of working directory ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.endpoint_fcn      import EndpointFCN
from utils.extract_endpoints import extract_endpoints_from_heatmap
from utils.pair_endpoints    import pair_endpoints
from utils.ocr_reader        import run_ocr_on_image
from utils.digit_regressor   import load_regressor, filter_ocr_by_regressor
from utils.scale_match       import match_lines_to_text, visualize_matches
from utils.scale_vote        import vote_scale, apply_scale, visualize_vote

# ── Default model paths ───────────────────────────────────────────────────────
DEFAULT_FCN_PATH       = os.path.join(PROJECT_ROOT, "models/scale_pipeline/endpoint_fcn_final.pth")
DEFAULT_REGRESSOR_PATH = os.path.join(PROJECT_ROOT, "models/scale_pipeline/digit_regressor.pth")

# ── Pipeline hyperparameters ──────────────────────────────────────────────────
HEATMAP_THRESHOLD = 0.32
NMS_KERNEL_SIZE   = 5
AXIS_TOLERANCE    = 10
MIN_LINE_LENGTH   = 20
MAX_LINE_LENGTH   = 480
MAX_MATCH_DIST    = 150


def run_pipeline(
    image_path:     str,
    out_dir:        str,
    use_regressor:  bool = True,
    save_vis:       bool = True,
    fcn_path:       str  = DEFAULT_FCN_PATH,
    regressor_path: str  = DEFAULT_REGRESSOR_PATH,
) -> dict:
    """
    Run the full dimension-extraction pipeline on one floor plan image.

    Returns a result dict:
    {
        "image":             str,
        "scale_px_per_unit": float,
        "unit_per_px":       float,
        "n_inliers":         int,
        "n_outliers":        int,
        "dimensions":        [{"line", "px_length", "text_value", "real_dimension", ...}, ...]
    }
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load FCN ───────────────────────────────────────────────────────────
    print(f"[1/7] Loading FCN from {fcn_path} …")
    if not os.path.exists(fcn_path):
        raise FileNotFoundError(f"FCN model not found: {fcn_path}\n"
                                "Run models/scale_pipeline/train_endpoint_fcn.py first.")
    model = EndpointFCN()
    model.load_state_dict(torch.load(fcn_path, map_location=device))
    model.to(device).eval()

    # ── 2. Inference → endpoints ──────────────────────────────────────────────
    print(f"[2/7] Running FCN on {os.path.basename(image_path)} …")
    pil_img = Image.open(image_path).convert("RGB")
    tensor  = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = model(tensor)

    endpoints = extract_endpoints_from_heatmap(
        heatmap,
        threshold=HEATMAP_THRESHOLD,
        kernel_size=NMS_KERNEL_SIZE,
    )
    print(f"         {len(endpoints)} endpoints extracted")

    # ── 3. Line pairing ───────────────────────────────────────────────────────
    print(f"[3/7] Pairing endpoints into dimension line segments …")
    pairs, unpaired = pair_endpoints(
        endpoints,
        axis_tolerance=AXIS_TOLERANCE,
        min_length=MIN_LINE_LENGTH,
        max_length=MAX_LINE_LENGTH,
    )
    print(f"         {len(pairs)} pairs  |  {len(unpaired)} unpaired endpoints")

    if not pairs:
        raise RuntimeError("No dimension line pairs found. "
                           "Try lowering HEATMAP_THRESHOLD or adjusting pairing parameters.")

    # ── 4. OCR ────────────────────────────────────────────────────────────────
    print(f"[4/7] Running EasyOCR …")
    import easyocr
    reader      = easyocr.Reader(["en"], gpu=(device.type == "cuda"))
    ocr_results = run_ocr_on_image(image_path, reader)
    print(f"         {len(ocr_results)} text detections")

    # ── 5. Digit-regressor double-check ───────────────────────────────────────
    if use_regressor and os.path.exists(regressor_path):
        print(f"[5/7] Applying digit-count regressor …")
        regressor   = load_regressor(regressor_path, device)
        image_bgr   = cv2.imread(image_path)
        ocr_results, rejected = filter_ocr_by_regressor(
            ocr_results, image_bgr, regressor, device
        )
        print(f"         kept {len(ocr_results)} / "
              f"{len(ocr_results) + len(rejected)} detections")
        if rejected:
            for r in rejected:
                print(f"         rejected: '{r.get('text','')}' — {r.get('reject_reason','')}")
    elif use_regressor:
        print(f"[5/7] Digit regressor not found at {regressor_path} — skipping.")
    else:
        print(f"[5/7] Digit regressor disabled — skipping.")

    # ── 6. Scale matching ─────────────────────────────────────────────────────
    print(f"[6/7] Matching lines to text labels …")
    matches, unmatched_lines, unmatched_texts = match_lines_to_text(pairs, ocr_results)
    print(f"         {len(matches)}/{len(pairs)} lines matched")

    if not matches:
        raise RuntimeError("No (line, text) matches found. "
                           "Check OCR results and MAX_MATCH_DIST.")

    if save_vis:
        vis_path = os.path.join(out_dir, "scale_matches.png")
        visualize_matches(image_path, matches, unmatched_lines, save_path=vis_path)

    # ── 7. K-Means scale voting ───────────────────────────────────────────────
    print(f"[7/7] Voting on consensus scale …")
    vote_result = vote_scale(matches)
    scale       = vote_result["scale_px_per_unit"]
    final_dims  = apply_scale(vote_result["inliers"], scale)

    print(f"\n{'='*55}")
    print(f"  Consensus scale: {scale:.6f} px/unit")
    print(f"  1 unit = {1/scale:.3f} px")
    print(f"  Inliers: {len(vote_result['inliers'])} / {len(matches)}")
    print(f"{'='*55}")
    for m in final_dims:
        (x1,y1),(x2,y2) = m["line"]
        orient = "V" if abs(x1-x2) < abs(y1-y2) else "H"
        print(f"  [{orient}] px={m['px_length']:.1f}  "
              f"text={m['text_value']}  → {m['real_dimension']:.2f} units")

    if save_vis:
        vote_vis = os.path.join(out_dir, "scale_vote.png")
        visualize_vote(vote_result, save_path=vote_vis)

    # ── Save JSON output ──────────────────────────────────────────────────────
    out = {
        "image":             image_path,
        "scale_px_per_unit": scale,
        "unit_per_px":       round(1.0 / scale, 6),
        "n_inliers":         len(vote_result["inliers"]),
        "n_outliers":        len(vote_result["outliers"]),
        "dimensions": [
            {
                "line":          [list(m["line"][0]), list(m["line"][1])],
                "px_length":     m["px_length"],
                "text_value":    m["text_value"],
                "real_dimension": m["real_dimension"],
            }
            for m in final_dims
        ],
    }
    out_json = os.path.join(out_dir, "dimensions.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_json}")

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract real-world dimensions from a floor plan image."
    )
    parser.add_argument("image", help="Path to floor plan image (PNG or JPG)")
    parser.add_argument(
        "--out-dir", default=os.path.join(PROJECT_ROOT, "output"),
        help="Directory to write results (default: ./output/)"
    )
    parser.add_argument(
        "--fcn-model", default=DEFAULT_FCN_PATH,
        help="Path to endpoint FCN weights (.pth)"
    )
    parser.add_argument(
        "--regressor-model", default=DEFAULT_REGRESSOR_PATH,
        help="Path to digit-count regressor weights (.pth)"
    )
    parser.add_argument(
        "--no-regressor", action="store_true",
        help="Skip the digit-count double-check filter"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip saving visualisation images"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    run_pipeline(
        image_path=args.image,
        out_dir=args.out_dir,
        use_regressor=not args.no_regressor,
        save_vis=not args.no_vis,
        fcn_path=args.fcn_model,
        regressor_path=args.regressor_model,
    )


if __name__ == "__main__":
    main()
