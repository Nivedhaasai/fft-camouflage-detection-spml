# Multi-Scale Texture Fusion for Camouflage Detection
## Frequency-Domain Analysis & Log-Polar Spectral Pooling

This project implements a robust, non-deep-learning approach to camouflage detection. Instead of relying on data-hungry neural networks, it leverages classical computer vision and signal processing—specifically **Fast Fourier Transforms (FFT)**, **Gabor Filter Banks**, and **Local Binary Patterns (LBP)**—to identify regions where the background texture has been disrupted by a foreground object.

### The Problem: Why FFT for Camouflage?
Camouflage works by matching the dominant features (color, orientation, and spatial frequency) of the environment. However, an object—even if perfectly color-matched—rarely matches the exact *periodic* structure of the background. 
- **What is FFT?** The Fast Fourier Transform decomposes an image into its constituent frequencies. While a natural background (like grass or rocks) has a specific statistical regularity in the frequency domain, a physical object introduces "edge noise" and phase shifts that show up as distinct patterns in the 2D power spectrum. 
- **Log-Polar Spectral Pooling:** By dividing the FFT power spectrum into 8 angular sectors (orientations) and 4 radial bands (frequencies), we create a "fingerprint" of the texture that is invariant to slight rotations—allowing us to detect the "off-beat" signature of a camouflaged animal or object.

---

### Technical Pipeline

#### 1. Multi-Scale Feature Extraction
The system processes the image at three different scales (32x32, 64x64, and 96x96 pixel patches). This captures both fine-grained skin/fur textures and larger structural disturbances.
- **Spectral Features:** Log-polar binned FFT coefficients.
- **Orientation Features:** Gabor filters at 0, 45, 90, and 135 degrees.
- **Spatial Regulator:** Uniform Local Binary Patterns (LBP) to describe micro-textures.

#### 2. Classification & Calibration
A **Support Vector Machine (SVM)** with an RBF kernel acts as the core classifier. 
- **Sigmoid Calibration:** We apply Platt scaling to convert raw SVM distances into true probabilities. This is critical for generating a heatmap that we can reliably threshold.
- **Threshold Optimization:** The system automatically calculates the optimal probability threshold (using F1-score maximization) to balance precision and recall.

#### 3. Spatial Morphology & CCA
Post-processing is applied to the probability map to ensure spatial consistency:
- **Connected Component Analysis (CCA):** Small noisy "detections" (smaller than 120 pixels) are filtered out to reduce false positives.
- **Bounding Box Generation:** Final detections are grouped into bounding boxes for intuitive visualization.

---

### Project Structure
- `src/fft_features.py`: The engine for signal processing and feature calculation.
- `train_fft_camouflage.py`: Dataset pipeline, Stratified 5-Fold Cross-Validation, and automated plotting.
- `predict_camouflage_map.py`: Inference script with spatial filtering and visualization.
- `models/`: Contains the `.joblib` model dict (includes weights + optimized threshold).
- `outputs/`: Performance plots (ROC, Confusion Matrix, Feature Importance) and detection results.

### Performance
Current benchmarks on the balanced dataset:
- **Accuracy:** ~74.4%
- **F1-Score:** ~0.76
- **Key Indicators:** Log-polar sector entropy and low-frequency spectral power are the highest-weighted features, confirming that frequency-domain analysis is the primary driver of detection.

### Installation & Usage
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python train_fft_camouflage.py
python predict_camouflage_map.py
```

---
*Developed as a study in Classical Machine Learning & Signal Processing.*
