import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import sys
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# IMPORT THE NEW CLASS
from utils.dataset import FloorplanHeatmapDataset

# Import your architecture


# --- CONFIGURATION ---
DATA_DIR = r"C:\Users\shres\OneDrive\Desktop\Python\wallDimensionProject\data\synthetic_dataset\images"       # Folder with Input PNGs
HEATMAP_DIR = r"C:\Users\shres\OneDrive\Desktop\Python\wallDimensionProject\data\labels_heatmap"      # Folder with Target Heatmap PNGs
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCHS = 5

import torch
import torch.nn as nn

class EndpointFCN(nn.Module):
    def __init__(self):
        super(EndpointFCN, self).__init__()
        
        # Contracting Path (Encoder)
        self.enc1 = self._conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 x 256
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 x 128
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 x 64
        
        # Bottleneck
        self.center = self._conv_block(256, 512)
        
        # Expansive Path (Decoder) - Using Deconv (ConvTranspose2d)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256) 
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output Layer: 1 Channel for Heatmap
        self.final = nn.Conv2d(64, 1, kernel_size=1) 
        self.sigmoid = nn.Sigmoid() 

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        c = self.center(p3)
        
        # Decoder with Skip Connections
        u3 = self.up3(c)
        cat3 = torch.cat([u3, e3], dim=1) 
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d3)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        
        return self.sigmoid(self.final(d1))

class DirectHeatmapDataset(Dataset):
    def __init__(self, img_dir, heatmap_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.heatmap_dir = heatmap_dir  
        
        # Transform for Input Image (RGB, Normalized)
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # output: [3, 512, 512], range 0.0 - 1.0
        ])
        
        # Transform for Heatmap (Grayscale, Normalized)
        # CRITICAL: We use Grayscale ('L') because heatmaps are 1-channel
        self.heatmap_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # output: [1, 512, 512], range 0.0 - 1.0
        ])
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Input Image
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path)
        
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.img_transform(image)
            
        # 2. Load Target Heatmap
        # Assuming heatmap has same filename as image
        heatmap_path = os.path.join(self.heatmap_dir, file_name)
        
        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Missing heatmap for {file_name} at {heatmap_path}")
            
        # Convert to Grayscale ('L') to ensure it is 1 channel
        heatmap = Image.open(heatmap_path).convert("L")
        heatmap_tensor = self.heatmap_transform(heatmap)
        
        # Return pair: (Input Image, Correct Answer)
        return image_tensor, heatmap_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Load Data
    dataset = FloorplanHeatmapDataset(DATA_DIR, HEATMAP_DIR, augment=True)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0  )
    print(f"Loaded {len(dataset)} training pairs.")
    
    # 2. Initialize Model
    model = EndpointFCN().to(device)
    
    # 3. Optimization
    # [cite_start]Paper uses Mean Squared Error [cite: 129] (implied for regression tasks)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    model.train()
    batch_count = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        print(f"Starting Epoch {epoch+1}") # <--- ADD THIS  
        
        for images, targets in dataloader:
            batch_count += 1
            print(f"  Processing Batch {batch_count}")
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            outputs = model(images)
            
            # Loss calculation
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"endpoint_fcn_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "endpoint_fcn_final.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train()