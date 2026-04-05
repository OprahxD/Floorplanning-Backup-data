import os
import json
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# YOLO Class IDs based on Phase A Rules
CLS_LOAD_WALL = 0
CLS_NONLOAD_WALL = 1
CLS_DOOR = 2
CLS_WINDOW = 3

def line_to_polygon(x1, y1, x2, y2, thickness, img_w, img_h):
    """Converts a line with thickness into a normalized 4-point polygon for YOLO."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    
    if length == 0: return None
    
    # Perpendicular vector for thickness
    px = -dy / length
    py = dx / length
    
    ht = thickness / 2.0
    
    # Calculate 4 corners of the rectangle
    corners = [
        (x1 + px * ht, y1 + py * ht),
        (x1 - px * ht, y1 - py * ht),
        (x2 - px * ht, y2 - py * ht),
        (x2 + px * ht, y2 + py * ht)
    ]
    
    # Normalize coordinates to 0.0 - 1.0 for YOLO
    norm_corners = []
    for cx, cy in corners:
        nx = max(0.0, min(1.0, cx / img_w))
        ny = max(0.0, min(1.0, cy / img_h))
        norm_corners.extend([f"{nx:.6f}", f"{ny:.6f}"])
        
    return " ".join(norm_corners)

def generate_yolo_segmentation(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(input_dir, 'jsons')
    if not os.path.exists(json_path): 
        return print(f"Error: {json_path} not found.")

    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    print(f"Generating YOLO polygon labels for {len(json_files)} files...")
    
    for j_file in json_files:
        with open(os.path.join(json_path, j_file)) as f: 
            data = json.load(f)
            
        img_path = os.path.join(input_dir, 'images', data['image_file'])
        # Read image to get actual dimensions for normalization
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # Helper: Flip Y-Axis (Matplotlib to Image coords)
        def fix_pt(pt):
            return (float(pt[0]), float(h - pt[1]))

        walls_by_id = {wall['id']: wall for wall in data.get('walls', [])}
        yolo_lines = []

        # 1. Walls
        if 'walls' in data:
            for wall in data['walls']:
                x1, y1 = fix_pt(wall['coords'][:2])
                x2, y2 = fix_pt(wall['coords'][2:])
                
                if wall.get('is_shared', False):
                    cls_id = CLS_NONLOAD_WALL
                    thickness = 6 # Thinner
                else:
                    cls_id = CLS_LOAD_WALL
                    thickness = 12 # Thicker
                
                poly_str = line_to_polygon(x1, y1, x2, y2, thickness, w, h)
                if poly_str:
                    yolo_lines.append(f"{cls_id} {poly_str}")

        # 2. Windows
        if 'windows' in data:
            for win in data['windows']:
                wx1, wy1 = fix_pt(win['coords'][:2])
                wx2, wy2 = fix_pt(win['coords'][2:])
                # Windows are slightly thicker than walls
                poly_str = line_to_polygon(wx1, wy1, wx2, wy2, 14, w, h)
                if poly_str:
                    yolo_lines.append(f"{CLS_WINDOW} {poly_str}")

        # 3. Doors (Using the Wall Cut logic)
        if 'doors' in data:
             for door in data['doors']:
                 if 'hinge' not in door or 'endpoint' not in door: continue
                 
                 hx, hy = door['hinge']
                 ex, ey = door['endpoint']
                 door_width = math.hypot(ex - hx, ey - hy)
                 
                 wall_id = door.get('wall_id')
                 if wall_id in walls_by_id:
                     w_data = walls_by_id[wall_id]
                     wx1, wy1, wx2, wy2 = w_data['coords']
                     
                     dx, dy = wx2 - wx1, wy2 - wy1
                     length = math.hypot(dx, dy) + 1e-5
                     dx, dy = dx / length, dy / length
                     
                     ox1 = hx - (dx * door_width * 0.5)
                     oy1 = hy - (dy * door_width * 0.5)
                     ox2 = hx + (dx * door_width * 0.5)
                     oy2 = hy + (dy * door_width * 0.5)
                     
                     p1 = fix_pt((ox1, oy1))
                     p2 = fix_pt((ox2, oy2))
                     
                     poly_str = line_to_polygon(p1[0], p1[1], p2[0], p2[1], 14, w, h)
                     if poly_str:
                         yolo_lines.append(f"{CLS_DOOR} {poly_str}")

        # Save to .txt file
        txt_filename = os.path.splitext(j_file)[0] + '.txt'
        with open(os.path.join(output_dir, txt_filename), 'w') as out_f:
            out_f.write("\n".join(yolo_lines))

    print("YOLO Segmentation Labels Done.")

# --- BUILT-IN VERIFICATION ---
if __name__ == "__main__":
    # Update these paths to match your local setup
    IN_DIR = "/home/oprah/Desktop/wallDimensionProject/Dataset"
    OUT_DIR = "/home/oprah/Desktop/wallDimensionProject/data/labels_yolo_seg"
    
    generate_yolo_segmentation(IN_DIR, OUT_DIR)
    
    # Verification: Draw the polygons from the generated text file
    txt_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.txt')]
    if txt_files:
        sample_txt = txt_files[0]
        img_name = sample_txt.replace('.txt', '.jpg')
        img_path = os.path.join(IN_DIR, 'images', img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            # Colors: Load (Blue), NonLoad (Black), Door (Green), Window (Red)
            colors = {
                0: (255, 0, 0),    # Blue (BGR in OpenCV)
                1: (0, 0, 0),      # Black
                2: (0, 255, 0),    # Green
                3: (0, 0, 255)     # Red
            }
            
            with open(os.path.join(OUT_DIR, sample_txt), 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                # Convert normalized coords back to absolute pixel values
                pts = [float(p) for p in parts[1:]]
                poly = np.array([
                    [int(pts[i] * w), int(pts[i+1] * h)] 
                    for i in range(0, len(pts), 2)
                ], np.int32)
                
                # Draw the polygon with transparency
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly], colors.get(cls_id, (255, 255, 255)))
                img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("YOLO Polygon Verification")
            plt.axis('off')
            plt.show()