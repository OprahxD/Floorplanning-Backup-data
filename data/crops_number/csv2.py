import cv2
import json
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

def process_number_verification(input_dir, output_dir, padding=5):
    crops_dir = os.path.join(output_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    
    json_path = os.path.join(input_dir, 'jsons')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    csv_path = os.path.join(output_dir, 'number_labels.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['filename', 'content', 'digit_count'])
    
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    print(f"Processing crops for {len(json_files)} files...")
    
    count = 0
    for j_file in json_files:
        with open(os.path.join(json_path, j_file)) as f: data = json.load(f)
            
        img_path = os.path.join(input_dir, 'images', data['image_file'])
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        if 'dimensions' in data:
            for i, dim in enumerate(data['dimensions']):
                x1, y1, x2, y2 = dim['bbox']
                
                # Padding
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
                
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Unique filename per crop
                crop_filename = f"{os.path.splitext(j_file)[0]}_crop_{i}.png"
                cv2.imwrite(os.path.join(crops_dir, crop_filename), crop)
                
                content = dim['val']
                digit_count = sum(c.isdigit() for c in content)
                
                writer.writerow([crop_filename, content, digit_count])
                count += 1
                
    csv_file.close()
    print(f"Done. Generated {count} text crops.")

# --- BUILT-IN VERIFICATION ---
if __name__ == "__main__":
    IN_DIR = "/home/oprah/Desktop/wallDimensionProject/Dataset"
    OUT_DIR = "/home/oprah/Desktop/wallDimensionProject/data/crops_number/output"
    
    
    # 1. Generate Data
    process_number_verification(IN_DIR, OUT_DIR)
    
    # 2. Verify Output
    print("\n--- VERIFICATION STEP ---")
    csv_path = os.path.join(OUT_DIR, 'number_labels.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            # Pick up to 5 random samples
            samples = df.sample(min(5, len(df)))
            
            plt.figure(figsize=(15, 4))
            for i, (_, row) in enumerate(samples.iterrows()):
                img_path = os.path.join(OUT_DIR, 'crops', row['filename'])
                if not os.path.exists(img_path): continue
                    
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(1, 5, i+1)
                plt.imshow(img)
                plt.title(f"Label: '{row['content']}'\nDigits: {row['digit_count']}")
                plt.axis('off')
            plt.show()
        else:
            print("CSV is empty.")
    else:
        print("CSV file not found.")
# claude --resume a6e3722d-c965-4f20-a177-5ef2b1271ffc 