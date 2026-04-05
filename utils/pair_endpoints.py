import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import sys
import os
import glob
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# 🎛️ HYPERPARAMETER CONFIGURATION
# ---------------------------------------------------------------------------
# Fine-tune these values based on how your FCN model is performing.
CONFIG = {
    # --- Endpoint Extraction (NMS) ---
    "heatmap_threshold": 0.32,    # Minimum confidence to be considered an endpoint (0.0 to 1.0)
    "nms_kernel_size": 5,        # Size of the local neighborhood for Non-Maximum Suppression (odd number)

    # --- Line Pairing Rules ---
    "axis_tolerance": 10,        # Max pixel deviation allowed for a line to still be considered straight
    "min_line_length": 20,       # Minimum pixel distance between two points to form a line (filters adjacent noise)
    "max_line_length": 480,      # Maximum pixel distance (filters points across opposite sides of the building)
    
    # --- Image Processing ---
    "target_resolution": 512,    # The resolution your FCN expects
}
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.extract_endpoints import extract_endpoints_from_heatmap

def pair_endpoints(endpoints, axis_tolerance, min_length, max_length):
    """
    Pairs detected endpoints into axis-aligned dimension line segments.
    """
    if len(endpoints) < 2:
        return [], list(endpoints)

    pts = np.array(endpoints, dtype=np.float32)
    n = len(pts)
    used = [False] * n

    # Build candidates: (alignment_error, distance, i, j)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(pts[i, 0] - pts[j, 0])
            dy = abs(pts[i, 1] - pts[j, 1])
            dist = float(np.hypot(dx, dy))

            if dist < min_length or dist > max_length:
                continue

            is_vertical   = dx <= axis_tolerance   
            is_horizontal = dy <= axis_tolerance   

            if not (is_vertical or is_horizontal):
                continue

            # Prioritize straightness over pure distance
            alignment_error = min(dx, dy)
            candidates.append((alignment_error, dist, i, j))

    # Sort primarily by straightness, then by shortest distance
    candidates.sort(key=lambda t: (t[0], t[1]))

    pairs = []
    for alignment_error, dist, i, j in candidates:
        if used[i] or used[j]:
            continue
        pairs.append((tuple(endpoints[i]), tuple(endpoints[j])))
        used[i] = True
        used[j] = True

    unpaired = [endpoints[i] for i in range(n) if not used[i]]
    return pairs, unpaired


def visualize_pairs(image_np, pairs, unpaired=None, save_path=None):
    """
    Draws paired dimension lines and unpaired endpoints on the image.
    """
    vis = image_np.copy()

    for (x1, y1), (x2, y2) in pairs:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.line(vis, p1, p2, color=(0, 255, 0), thickness=2)
        cv2.circle(vis, p1, radius=4, color=(0, 200, 255), thickness=-1)
        cv2.circle(vis, p2, radius=4, color=(0, 200, 255), thickness=-1)

    if unpaired:
        for (x, y) in unpaired:
            cv2.circle(vis, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Line Pairing | {len(pairs)} lines | {len(unpaired or [])} unpaired")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
    MODEL_PATH   = os.path.join(PROJECT_ROOT, "models/scale_pipeline/endpoint_fcn_final.pth")
    IMAGE_DIR    = os.path.join(PROJECT_ROOT, "Dataset/images")

    # --- Load model ---
    sys.path.insert(0, PROJECT_ROOT)
    from utils.endpoint_fcn import EndpointFCN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EndpointFCN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"Model loaded on {device}")

    # --- Pick first image ---
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    if not img_paths:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    img_path = img_paths[0]
    print(f"Running on: {img_path}")

    # --- Preprocess ---
    pil_img = Image.open(img_path).convert("RGB")
    res = CONFIG["target_resolution"]
    orig_np = cv2.resize(np.array(pil_img), (res, res)) 

    tensor = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
    ])(pil_img).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        heatmap = model(tensor) 

    heatmap_np = heatmap[0, 0].cpu().numpy()
    print(f"Heatmap stats — min: {heatmap_np.min():.4f}  max: {heatmap_np.max():.4f}  mean: {heatmap_np.mean():.4f}")

    # --- Extract endpoints (Using Config) ---
    endpoints = extract_endpoints_from_heatmap(
        heatmap, 
        threshold=CONFIG["heatmap_threshold"], 
        kernel_size=CONFIG["nms_kernel_size"]
    )
    print(f"Extracted {len(endpoints)} endpoints")

    # --- Pair endpoints (Using Config) ---
    pairs, unpaired = pair_endpoints(
        endpoints, 
        axis_tolerance=CONFIG["axis_tolerance"],
        min_length=CONFIG["min_line_length"],
        max_length=CONFIG["max_line_length"]
    )
    print(f"Paired into {len(pairs)} dimension lines ({len(unpaired)} unpaired)")

    for i, ((x1, y1), (x2, y2)) in enumerate(pairs):
        length = np.hypot(x2 - x1, y2 - y1)
        orientation = "V" if abs(x1 - x2) < abs(y1 - y2) else "H"
        print(f"  Line {i+1:3d}: ({x1},{y1}) -> ({x2},{y2}) len={length:.1f}px [{orientation}]")

    # --- Visualise ---
    bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    visualize_pairs(bgr, pairs, unpaired)