import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class FloorplanHeatmapDataset(Dataset):
    """
    A PyTorch Dataset that loads Input Floorplans and their corresponding
    Target Heatmaps for the Endpoint Detection Network.
    """
    def __init__(self, img_dir, heatmap_dir, augment=False):
        """
        Args:
            img_dir (str): Path to folder containing source PNG images.
            heatmap_dir (str): Path to folder containing target heatmap PNGs.
            augment (bool): If True, applies random rotations/noise (Paper technique).
        """
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.augment = augment
        
        # Get list of images Support jpg, jpeg, and png
        extensions = ["*.jpg", "*.jpeg", "*.png"]
        self.img_paths = []
        for ext in extensions:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths = sorted(self.img_paths)
        
        # Standardize size as per paper 
        self.target_size = (512, 512)

    def __len__(self):
        return len(self.img_paths)

    def transform(self, image, heatmap):
        # 1. Resize
        # Use bilinear for images (smooth)
        image = TF.resize(image, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        # Use bilinear for heatmaps too since they are gaussian blobs, not integer masks
        heatmap = TF.resize(heatmap, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)

        # 2. Augmentation (Optional - as per paper [cite: 421])
        if self.augment:
            # Random Rotation (-10 to 10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                heatmap = TF.rotate(heatmap, angle)

        # 3. To Tensor (Normalizes 0-255 to 0.0-1.0)
        image = TF.to_tensor(image)
        heatmap = TF.to_tensor(heatmap)

        return image, heatmap

    def __getitem__(self, idx):
        # 1. Get Input Image Path
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path)
        
        # 2. Construct Heatmap Path (Robust Check)
        # Extract the filename without extension (e.g., "consolidated_plan_1")
        base_name = os.path.splitext(file_name)[0]
        
        # Priority 1: Check for .png (Standard for masks/heatmaps)
        heatmap_path = os.path.join(self.heatmap_dir, base_name + ".png")
        
        # Priority 2: Check for .jpg (If you saved them as jpg)
        if not os.path.exists(heatmap_path):
            heatmap_path = os.path.join(self.heatmap_dir, base_name + ".jpg")
            
        # Priority 3: Check for exact filename match (fallback)
        if not os.path.exists(heatmap_path):
            heatmap_path = os.path.join(self.heatmap_dir, file_name)

        # 3. Validation
        if not os.path.exists(heatmap_path):
            # Detailed error message to help you debug
            raise FileNotFoundError(
                f"\n❌ CRITICAL ERROR:\n"
                f"   Input Image: {file_name}\n"
                f"   Looking for heatmap at: {heatmap_path}\n"
                f"   Folder Checked: {self.heatmap_dir}\n"
                f"   PLEASE CHECK: Does the file exist? Is it .png or .jpg?"
            )

        # 4. Load Images
        image = Image.open(img_path).convert("RGB")
        
        # Load Heatmap & Convert to Grayscale
        heatmap = Image.open(heatmap_path).convert("L")
        
        # 5. Apply Transforms
        x, y = self.transform(image, heatmap)

        return x, y

# --- Quick Test Block ---
if __name__ == "__main__":
    # Create dummy folders to test the logic if you run this file directly
    print("Testing Dataset Logic...")
    
    # 1. Create dummy data
    os.makedirs("temp_test_img", exist_ok=True)
    os.makedirs("temp_test_hm", exist_ok=True)
    
    # Create a white image and a black heatmap
    Image.new('RGB', (1000, 1000), 'white').save("temp_test_img/sample.png")
    Image.new('L', (1000, 1000), 'black').save("temp_test_hm/sample.png")
    
    # 2. Init Dataset
    ds = FloorplanHeatmapDataset("temp_test_img", "temp_test_hm", augment=True)
    
    # 3. Get Item
    img_t, hm_t = ds[0]
    
    print(f"Input Shape: {img_t.shape} (Expected: 3, 512, 512)")
    print(f"Target Shape: {hm_t.shape} (Expected: 1, 512, 512)")
    print(f"Values Normalized? Max val: {img_t.max()}")
    
    # Cleanup
    import shutil
    shutil.rmtree("temp_test_img")
    shutil.rmtree("temp_test_hm")
    print("✅ Test Passed.")