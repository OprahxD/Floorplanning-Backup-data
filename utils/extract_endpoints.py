import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_endpoints_from_heatmap(heatmap_tensor, threshold=0.5, kernel_size=5):
    """
    Extracts exact (x, y) coordinates of local peaks from a heatmap tensor.
    
    Args:
        heatmap_tensor (torch.Tensor): The output from EndpointFCN (1, 1, 512, 512).
        threshold (float): Minimum confidence score (0.0 to 1.0) to be considered an endpoint.
        kernel_size (int): Size of the local neighborhood (must be an odd number).
                           Larger = points must be further apart.
                           
    Returns:
        list of tuples: [(x1, y1), (x2, y2), ...]
    """
    # 1. Format Check: Ensure tensor is (Batch, Channel, Height, Width)
    if heatmap_tensor.dim() == 2:
        heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)
    elif heatmap_tensor.dim() == 3:
        heatmap_tensor = heatmap_tensor.unsqueeze(0)

    # 2. Apply Max Pooling (The NMS logic)
    # By using stride=1 and padding=kernel//2, the output size stays exactly 512x512
    pad = kernel_size // 2
    max_pooled = F.max_pool2d(heatmap_tensor, kernel_size=kernel_size, stride=1, padding=pad)
    
    # 3. Find the Peaks
    # A pixel is a peak ONLY IF its value didn't change after max pooling (meaning it WAS the max)
    # AND it is above our confidence threshold.
    is_peak = (heatmap_tensor == max_pooled) & (heatmap_tensor > threshold)
    
    # 4. Extract Coordinates
    # torch.nonzero returns the indices of all True values.
    # For a (1, 1, 512, 512) tensor, it returns [batch, channel, y, x]
    peaks = torch.nonzero(is_peak[0, 0])
    
    # Convert to standard Python list of (x, y) tuples
    # Note: PyTorch indices are (row, col) which is (y, x). We swap them for image plotting.
    coordinates = [(int(x), int(y)) for y, x in peaks]
    
    return coordinates


# --- TEST AND VISUALIZATION BLOCK ---
if __name__ == "__main__":
    print("Testing Max Pooling Extraction...")
    
    # 1. Create a dummy heatmap (512x512) of all zeros
    dummy_heatmap = torch.zeros((1, 1, 512, 512))
    
    # 2. Inject fake "clouds" of probability (like the model would output)
    # Cloud 1 at (x=100, y=150)
    dummy_heatmap[0, 0, 149:152, 99:102] = 0.8  # surrounding pixels
    dummy_heatmap[0, 0, 150, 100] = 0.95        # THE PEAK
    
    # Cloud 2 at (x=400, y=300)
    dummy_heatmap[0, 0, 298:303, 398:403] = 0.7 
    dummy_heatmap[0, 0, 300, 400] = 0.88        # THE PEAK
    
    # Noise (Below threshold, should be ignored)
    dummy_heatmap[0, 0, 50, 50] = 0.4
    
    # 3. Run our extraction function
    # We want points above 0.5 confidence, looking in a 5x5 pixel window
    coords = extract_endpoints_from_heatmap(dummy_heatmap, threshold=0.5, kernel_size=5)
    
    print(f"Extracted {len(coords)} endpoints:")
    for i, (x, y) in enumerate(coords):
        confidence = dummy_heatmap[0, 0, y, x].item()
        print(f"  Point {i+1}: (x={x}, y={y}) - Confidence: {confidence:.2f}")

    # 4. Visualize it
    heatmap_np = dummy_heatmap[0, 0].numpy()
    
    plt.figure(figsize=(8, 8))
    plt.title("Extracted Endpoints from Heatmap")
    plt.imshow(heatmap_np, cmap='magma')
    
    # Plot red 'X' over the extracted coordinates
    for x, y in coords:
        plt.plot(x, y, 'rx', markersize=15, markeredgewidth=2)
        
    plt.axis('off')
    plt.show()