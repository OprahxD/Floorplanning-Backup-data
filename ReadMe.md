# Floor Plan Recognition & Reconstruction Project

This repository houses a robust, multi-stage machine learning pipeline designed to automatically recognize, measure, and reconstruct residential floor plans[cite: 1, 2, 3]. 



## 📖 Our Philosophy: "Redundancy for Reliability"


We prioritize cross-verification and global consensus over single-point predictions:
* **Decoupled Detection:** We do not detect "dimension lines" directly. Instead, we treat texts and line endpoints as separate entities[cite: 138]. We use a Fully Convolutional Network (FCN) to predict endpoints as probability heatmaps rather than bounding boxes[cite: 200, 201].
* **The "Double Check" OCR:** We never trust the OCR output alone. We utilize a double inspection mechanism[cite: 143]. A separate regression network counts the quantity of digits in a number area[cite: 139, 142]. If the OCR reads "3040" but the regressor sees 3 digits, the result is discarded to prevent massive scaling errors[cite: 144, 145].
* **Global Voting over Local Linking:** We do not rigidly link one dimension line to one specific wall. Instead, we match numbers to lines to calculate candidate scales[cite: 195, 196]. We then run K-Means clustering on all these results to vote on a single, global pixel-to-meter ratio, automatically eliminating outliers[cite: 196, 197, 199].
* **Adaptive Wall Thickness:** Rather than assuming fixed pixel wall widths across different image resolutions, we extract the bounding boxes of doors and windows[cite: 216, 218]. We use the median of their widths to dynamically calibrate the wall thickness[cite: 225].

---

## 🚀 What We Have Built Until Now

We have successfully mapped out the architecture and completed the foundational engineering for **Phase B (Scale Calculation & Line Detection)**[cite: 137, 138], along with specifying the requirements for **Phase A (Structural Segmentation)**.

### 1. Model Architecture
* **`models/scale_pipeline/endpoint_fcn.py`**: Built the Fully Convolutional Network (FCN) with deconvolution layers[cite: 200]. This model takes an input image and outputs a heatmap showing the exact locations of dimension line endpoints[cite: 201].

### 2. Data Pipeline & Processing
* **`utils/preprocess.py`**: Created a standardized transformation pipeline to load floor plans, convert them to RGB, resize them to the required 512x512 resolution[cite: 281, 431], and normalize them into PyTorch tensors.
* **`utils/dataset.py`**: Engineered a robust `DataLoader` class (`FloorplanHeatmapDataset`) that pairs our synthetic floor plans with their target heatmaps[cite: 432]. It includes built-in data augmentation (random rotations and resizing)[cite: 420, 421, 422].
* **`utils/extract_endpoints.py`**: Implemented a Max Pooling algorithm to extract the exact endpoints from the heatmaps[cite: 439, 440].

### 3. Training & Validation
* **`models/scale_pipeline/train_endpoint_fcn.py`**: Wrote the complete training loop for the endpoints detection model using a batch size of 8 and 500 epochs[cite: 432].
* **`utils/check_ground_truth.py`**: Built a sanity-check script that reads JSON labels and dynamically generates Gaussian heatmaps (like the labels shown in Figure 14) to verify our synthetic data aligns perfectly[cite: 414, 432].
* **`models/scale_pipeline/visualize_results.py`**: Created an inference and visualization tool to display the Original Image, Ground Truth, Model Prediction, and a blended overlay.

### 4. Phase A (Structure) Specifications
* Drafted the exact input/output deliverables for the semantic segmentation model, defining the strict class IDs required: Load-bearing walls (blue) [cite: 9], Non-load-bearing walls (black) [cite: 9], Doors (green) [cite: 11], and Windows (red)[cite: 11].