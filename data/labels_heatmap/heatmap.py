import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def generate_heatmap(width, height, points, sigma=3):
    """ 
    Generates a grayscale heatmap.
    width, height: The TARGET resolution (512, 512)
    points: List of (x, y) tuples already scaled to this resolution.
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for x, y in points:
        # Check bounds
        if x < 0 or x >= width or y < 0 or y >= height: continue
            
        roi_size = sigma * 3
        x_min = max(0, int(x - roi_size))
        x_max = min(width, int(x + roi_size + 1))
        y_min = max(0, int(y - roi_size))
        y_max = min(height, int(y + roi_size + 1))
        
        grid_y, grid_x = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (grid_x - x)**2 + (grid_y - y)**2
        blob = np.exp(-dist_sq / (2 * sigma**2))
        
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], blob)
        
    return heatmap

def process_scale_calibration(input_dir, output_dir, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(input_dir, 'jsons')
    if not os.path.exists(json_path): return print(f"Error: {json_path} not found.")

    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    print(f"Generating 512x512 heatmaps for {len(json_files)} files...")
    
    for j_file in json_files:
        with open(os.path.join(json_path, j_file)) as f: data = json.load(f)
            
        img_path = os.path.join(input_dir, 'images', data['image_file'])
        # We only need dimensions, avoiding full imread speeds this up
        # But to be safe, let's load to get exact W/H
        img = cv2.imread(img_path)
        if img is None: continue
        orig_h, orig_w = img.shape[:2]
        
        # Calculate Scale Factors
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        
        endpoints = []
        if 'dimensions' in data:
            for dim in data['dimensions']:
                if 'dim_line' in dim:
                    raw_x1, raw_y1, raw_x2, raw_y2 = dim['dim_line']
                    
                    # 1. Flip Y (Matplotlib -> Image coords)
                    y1_img = orig_h - raw_y1
                    y2_img = orig_h - raw_y2
                    
                    # 2. Resize (Image coords -> 512x512 coords)
                    # We map the point to the new small resolution
                    px1 = raw_x1 * scale_x
                    py1 = y1_img * scale_y
                    px2 = raw_x2 * scale_x
                    py2 = y2_img * scale_y
                    
                    endpoints.append((px1, py1))
                    endpoints.append((px2, py2))
        
        # Generate on SMALL canvas
        # Note: We reduce sigma slightly because the image is smaller
        heatmap = generate_heatmap(target_size[0], target_size[1], endpoints, sigma=3)
        
        # Save
        heatmap_img = (heatmap * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, os.path.splitext(j_file)[0] + '.png')
        cv2.imwrite(output_path, heatmap_img)
    
    print("Done.")

# --- VERIFICATION ---
if __name__ == "__main__":
    IN_DIR = "/home/oprah/Desktop/wallDimensionProject/Dataset"
    OUT_DIR = "/home/oprah/Desktop/wallDimensionProject/data/labels_heatmap"
    
    process_scale_calibration(IN_DIR, OUT_DIR, target_size=(512, 512))
    
    # Verify
    files = [f for f in os.listdir(OUT_DIR) if f.endswith('_heatmap.png')]
    if files:
        sample_file = files[0]
        heatmap = cv2.imread(os.path.join(OUT_DIR, sample_file), cv2.IMREAD_GRAYSCALE)
        print(f"Output Shape: {heatmap.shape} (Should be 512, 512)")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap='hot')
        plt.title(f"512x512 Heatmap Check")
        plt.axis('off')
        plt.show()