import cv2
import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt

# Class Definitions
CLASS_BG = 0
CLASS_WALL_LOAD = 1      # Red
CLASS_WALL_NONLOAD = 2   # Blue
CLASS_DOOR = 3           # Green (Opening)
CLASS_WINDOW = 4         # Yellow

def process_structure_segmentation(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(input_dir, 'jsons')
    if not os.path.exists(json_path): return print(f"Error: {json_path} not found.")

    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    print(f"Generating masks with door cuts for {len(json_files)} files...")
    
    for j_file in json_files:
        with open(os.path.join(json_path, j_file)) as f: data = json.load(f)
            
        img_path = os.path.join(input_dir, 'images', data['image_file'])
        src_img = cv2.imread(img_path)
        if src_img is None: continue
        h, w = src_img.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Helper: Flip Y-Axis for OpenCV
        def fix_pt(pt):
            return (int(pt[0]), int(h - pt[1]))

        # 1. Index Walls by ID so we can look them up later
        walls_by_id = {w['id']: w for w in data.get('walls', [])}

        # 2. Draw Walls First
        if 'walls' in data:
            for wall in data['walls']:
                x1, y1 = fix_pt(wall['coords'][:2])
                x2, y2 = fix_pt(wall['coords'][2:])
                
                if wall.get('is_shared', False):
                    wall_class = CLASS_WALL_NONLOAD
                    thickness = 3
                else:
                    wall_class = CLASS_WALL_LOAD
                    thickness = 5
                
                cv2.line(mask, (x1, y1), (x2, y2), wall_class, thickness)

        # 3. Draw Windows (Overwrite Walls)
        if 'windows' in data:
            for win in data['windows']:
                wx1, wy1 = fix_pt(win['coords'][:2])
                wx2, wy2 = fix_pt(win['coords'][2:])
                cv2.line(mask, (wx1, wy1), (wx2, wy2), CLASS_WINDOW, 5)

        # 4. Draw Doors (Overwrite Walls - THE FIX)
        if 'doors' in data:
             for door in data['doors']:
                 if 'hinge' not in door or 'endpoint' not in door: continue
                 
                 # A. Get Door Geometry
                 hx, hy = door['hinge']     # Hinge (on the wall)
                 ex, ey = door['endpoint']  # Leaf Tip
                 
                 # Calculate Door Width (Distance from hinge to tip)
                 door_width = math.hypot(ex - hx, ey - hy)
                 
                 # B. Find the Linked Wall Vector to draw the Opening along it
                 wall_id = door.get('wall_id')
                 if wall_id in walls_by_id:
                     w_data = walls_by_id[wall_id]
                     wx1, wy1, wx2, wy2 = w_data['coords']
                     
                     # Vector of the wall
                     dx, dy = wx2 - wx1, wy2 - wy1
                     length = math.hypot(dx, dy) + 1e-5
                     
                     # Normalize
                     dx, dy = dx / length, dy / length
                     
                     # C. Draw "Opening" Line on the Wall
                     # We draw a line centered at the hinge, along the wall direction
                     # Length = door_width
                     
                     # Start point (Hinge - half width)
                     # End point (Hinge + half width)
                     # Note: In reality, doors swing from one side, but centered ensures we cut the wall effectively.
                     
                     ox1 = hx - (dx * door_width * 0.5)
                     oy1 = hy - (dy * door_width * 0.5)
                     ox2 = hx + (dx * door_width * 0.5)
                     oy2 = hy + (dy * door_width * 0.5)
                     
                     p1 = fix_pt((ox1, oy1))
                     p2 = fix_pt((ox2, oy2))
                     
                     # Draw the OPENING (This cuts the red/blue wall line)
                     cv2.line(mask, p1, p2, CLASS_DOOR, 5) 
                     
                     # D. Optional: Draw the Leaf sticking out (Visual aid, usually optional for segmentation)
                     # p_hinge = fix_pt((hx, hy))
                     # p_tip = fix_pt((ex, ey))
                     # cv2.line(mask, p_hinge, p_tip, CLASS_DOOR, 3)

        # Save
        output_path = os.path.join(output_dir, os.path.splitext(j_file)[0] + '_mask.png')
        cv2.imwrite(output_path, mask)
    
    print("Done.")

# --- VERIFICATION ---
if __name__ == "__main__":
    IN_DIR = r"C:\Users\shres\OneDrive\Desktop\Python\syntheticDataGeneration\data\dataset_consolidated"
    OUT_DIR = r"C:\Users\shres\OneDrive\Desktop\Python\syntheticDataGeneration\measurements\complimentary\mask" 
    
    process_structure_segmentation(IN_DIR, OUT_DIR)
    
    # Verify
    print("\n--- CHECKING DOORS ---")
    files = [f for f in os.listdir(OUT_DIR) if f.endswith('_mask.png')]
    if files:
        sample_file = files[0]
        original_name = sample_file.replace('_mask.png', '.jpg')
        
        orig_img = cv2.imread(os.path.join(IN_DIR, 'images', original_name))
        mask = cv2.imread(os.path.join(OUT_DIR, sample_file), cv2.IMREAD_GRAYSCALE)
        
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 0, 0]   # Red
        color_mask[mask == 2] = [0, 0, 255]   # Blue
        color_mask[mask == 3] = [0, 255, 0]   # Green (Door Opening)
        color_mask[mask == 4] = [255, 255, 0] # Yellow
        
        overlay = cv2.addWeighted(orig_img, 0.7, color_mask, 0.3, 0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Fixed: Green Door 'Cuts' the Red Wall")
        plt.axis('off')
        plt.show()
    