import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

# --- PATH SETUP ---
# Add project root to path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.endpoint_fcn import EndpointFCN
from utils.dataset import FloorplanHeatmapDataset

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\shres\OneDrive\Desktop\Python\endpoint_fcn_final.pth"  # Replace with your latest checkpoint
DATA_DIR = r"C:\Users\shres\OneDrive\Desktop\Python\wallDimensionProject\data\synthetic_dataset\images"
HEATMAP_DIR = r"C:\Users\shres\OneDrive\Desktop\Python\wallDimensionProject\data\labels_heatmap"

def visualize_prediction(model_path, img_index=0):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing using: {device}")

    # 2. Load the Trained Model
    model = EndpointFCN().to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}")
        print("   Have you trained the model yet?")
        return

    # Load weights (handle CPU/GPU mapping)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode (important!)

    # 3. Load a Sample from Dataset
    # We use the dataset class to ensure transforms are identical to training
    dataset = FloorplanHeatmapDataset(DATA_DIR, HEATMAP_DIR, augment=False)
    
    if img_index >= len(dataset):
        print(f"Index {img_index} out of range. Using random index.")
        img_index = np.random.randint(0, len(dataset)-1)
        
    # Get tensor pair
    input_tensor, target_tensor = dataset[img_index]
    
    # Add batch dimension (3, 512, 512) -> (1, 3, 512, 512)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 4. Run Inference
    with torch.no_grad():
        output_batch = model(input_batch)
    
    # 5. Process for Visualization
    # Remove batch dim and convert to numpy
    # Input Image (RGB)
    img_np = input_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Ground Truth Heatmap (Grayscale)
    target_np = target_tensor.squeeze().cpu().numpy()
    
    # Predicted Heatmap (Grayscale)
    pred_np = output_batch.squeeze().cpu().numpy()

    # 6. Create Overlays (Optional but helpful)
    # Resize prediction to match if necessary (it should already match)
    pred_uint8 = (pred_np * 255).astype(np.uint8)
    target_uint8 = (target_np * 255).astype(np.uint8)
    
    # Apply colormap (Jet = Blue to Red)
    pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
    pred_color = cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB)
    
    # Blend with original (need original in uint8 0-255)
    img_uint8 = (img_np * 255).astype(np.uint8).copy() # Copy to avoid read-only errors
    overlay = cv2.addWeighted(img_uint8, 0.6, pred_color, 0.4, 0)

    # 7. Plotting
    plt.figure(figsize=(20, 5))

    # A. Original Image
    plt.subplot(1, 4, 1)
    plt.title("Original Input")
    plt.imshow(img_np)
    plt.axis('off')

    # B. Ground Truth (What it should be)
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth Heatmap")
    plt.imshow(target_np, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # C. Prediction (What the model thinks)
    plt.subplot(1, 4, 3)
    plt.title("Model Prediction")
    plt.imshow(pred_np, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # D. Overlay (Context)
    plt.subplot(1, 4, 4)
    plt.title("Prediction Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("✅ Visualization complete.")

if __name__ == "__main__":
    # Change 'img_index' to look at different files (0, 1, 2...)
    visualize_prediction(MODEL_PATH, img_index=5)