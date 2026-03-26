Camouflage Detection with FFT Texture Features

Overview
This project provides a complete signal-processing + classical machine learning pipeline for detecting camouflaged regions using FFT-based frequency-domain texture analysis.

Data layout
Expected directory structure:
- dataset/camo
- dataset/non_camo
- masks

Mask matching rule:
- Every image in dataset/camo must have a mask with the same filename stem in masks.
- Images in dataset/non_camo are treated as background-only samples.

Method
1. Split images into 64x64 patches.
2. Auto-label camo patches using mask_patch mean intensity:
   - mean > 0.1 -> camouflage
   - otherwise -> background
3. Label all non_camo patches as background.
3. Compute FFT patch features:
   - Mean spectral energy
   - Standard deviation
   - Shannon entropy
   - High-frequency energy ratio
   - Radial frequency distribution
   - Horizontal directional energy
   - Vertical directional energy
4. Normalize features using StandardScaler.
5. Train an SVM classifier with RBF kernel.
6. Predict camouflage patches and generate probability heatmaps.

Setup
1. Create and activate a Python environment.
2. Install dependencies:
   pip install -r requirements.txt

Training
Run:
python train_fft_camouflage.py

Model artifact:
- models/svm_fft_camouflage_model.joblib

Inference
Run:
python predict_camouflage_map.py --image dataset/camo/<your_image>.jpg --output outputs/<your_image>_pred.png

Inference output includes:
- original image
- FFT magnitude spectrum
- probability heatmap
- detected camouflage overlay with bounding boxes
