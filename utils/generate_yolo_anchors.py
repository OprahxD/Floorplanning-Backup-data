import numpy as np
import json
import glob
import os
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
LABEL_DIR = "../../Dataset/jsons/"  # Where your JSONs are
NUM_ANCHORS = 5                             # Paper specifically requires 5 [cite: 289]
IMAGE_SIZE = 608                            # Standardizing to the testing size [cite: 425]

def load_all_bounding_boxes(label_dir):
    """
    Scans your JSON labels and extracts the width and height of every number box.
    Returns an array of [width, height] normalized to the IMAGE_SIZE.
    """
    json_paths = glob.glob(os.path.join(label_dir, "*.json"))
    boxes = []
    
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            
            # CUSTOMIZE: Extract bounding boxes based on your JSON format
            # Assuming format has 'shapes' with 'label' == 'number' or 'text'
            if "shapes" in data:
                for shape in data["shapes"]:
                    # Adjust this label check to match your synthetic data
                    if shape.get("label") in ["number", "text", "dimension_text"]:
                        pts = shape["points"]
                        # Calculate width and height
                        # Assuming points are [[x1, y1], [x2, y2]]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        
                        w = max(xs) - min(xs)
                        h = max(ys) - min(ys)
                        
                        # Normalize against your original synthetic image size
                        # (Assuming original images were generated at 1000x1000 for example, 
                        # you need to scale them to the YOLO 608x608 size)
                        # Let's assume they are already 608x608 for this math, or adjust accordingly.
                        boxes.append([w, h])
                        
    return np.array(boxes)

def main():
    print("Loading bounding boxes...")
    boxes = load_all_bounding_boxes(LABEL_DIR)
    
    if len(boxes) == 0:
        print("❌ No bounding boxes found. Check your JSON parsing logic.")
        return
        
    print(f"Found {len(boxes)} text bounding boxes. Running K-Means++...")
    
    # Run K-Means++ clustering
    kmeans = KMeans(n_clusters=NUM_ANCHORS, init='k-means++', random_state=42)
    kmeans.fit(boxes)
    
    # Get the cluster centers (these are your anchors)
    anchors = kmeans.cluster_centers_
    
    # Sort them by area (width * height)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    
    print("\n✅ Custom Anchors Calculated (Width, Height):")
    formatted_anchors = []
    for w, h in anchors:
        print(f"  [{int(w)}, {int(h)}]")
        formatted_anchors.append(f"{int(w)},{int(h)}")
        
    print("\nYOLO Configuration String:")
    print("anchors = " + ",  ".join(formatted_anchors))

if __name__ == "__main__":
    main()