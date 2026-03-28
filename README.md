# Camouflage Detection via Signal Processing & Deep Learning
## Multi-Scale Texture Fusion: FFT, Wavelets, and YOLOv11

This project explores camouflage detection through the lens of **Signal Processing for Machine Learning (SPML)**. While modern deep learning provides high accuracy, signal processing techniques like the **Fast Fourier Transform (FFT)** and **Discrete Wavelet Transform (DWT)** provide the mathematical foundation for understanding how camouflage disrupts the periodic and multi-resolution structure of natural backgrounds.

---

### 1. The Core: Signal Processing Foundation

Camouflage is designed to match the dominant spatial frequencies and orientations of an environment. Our project identifies these targets by analyzing where these patterns break down.

#### A. Fast Fourier Transform (FFT) - Frequency Domain Analysis
- **Theory**: We decompose image patches into their constituent 2D frequencies. Natural backgrounds (grass, sand, rock) typically have a consistent statistical regularity in the frequency domain.
- **Implementation**: We use **Log-Polar Spectral Pooling**. By dividing the FFT power spectrum into 8 angular sectors and 4 radial frequency bands, we create a "spectral fingerprint." 
- **Detection**: A camouflaged object—even if color-matched—introduces "edge noise" and phase shifts that create anomalies in these frequency bins, allowing a trained SVM to flag the region.

#### B. Discrete Wavelet Transform (DWT) - Multi-Resolution Analysis
- **Theory**: Unlike FFT, which only provides frequency information, Wavelets provide both **Time-Frequency (Spatial-Frequency)** localization. This is crucial for pinpointing *where* a texture change occurs.
- **Implementation**: We apply 2D Wavelet decomposition (e.g., Daubechies or Haar) to extract:
  - **LL (Approximation)**: Low-frequency structural data.
  - **LH, HL, HH (Details)**: Horizontal, vertical, and diagonal high-frequency details.
- **Detection**: Camouflage often fails to match the high-frequency "detail" coefficients of the background at multiple scales. We analyze the energy distribution across these wavelet sub-bands to identify artificial boundaries.

---

### 2. High-Accuracy Refinement: YOLOv11 Segmentation

To bridge classical SPML with state-of-the-art AI, we implemented a **YOLOv11s-Segmentation** model. 
- **Role**: While FFT and Wavelets provide the "why" (mathematical disruption), YOLOv11 provides the "where" with pixel-perfect precision.
- **Training**: Optimized for small datasets (~200 samples) using heavy spatial augmentations (flipping, scaling, and HSV translation).
- **Execution**: Runs on an **NVIDIA RTX A2000 GPU** with Mixed Precision (AMP) for real-time inference.

---

### 3. Project Pipeline

1.  **Data Preprocessing**: Merging heterogeneous camouflage datasets and converting binary masks into YOLO-compatible polygons.
2.  **Feature Fusion**: Extracting LBP (Local Binary Patterns), Gabor filters (orientation), and FFT/Wavelet coefficients.
3.  **Hybrid Inference**:
    - **SPML**: FFT/SVM analysis for texture-based anomaly detection.
    - **Deep Learning**: YOLOv11 for high-accuracy instance segmentation.

---

### 4. Technical Specifications & Setup

#### Environment
- **Python**: 3.11 (optimized for CUDA 12.1)
- **Hardware**: NVIDIA RTX A2000 Laptop GPU
- **Key Libraries**: `ultralytics`, `torch`, `opencv-python`, `scikit-learn`, `flask`.

#### Execution
To run the High-Accuracy UI (GPU required):
```powershell
# Activate the optimized 3.11 environment
& ".\.venv_yolo311\Scripts\activate.ps1"

# Launch the Web Dashboard
python app.py
```

#### Files
- `src/fft_features.py`: Frequency domain extraction logic.
- `predict_camouflage_yolo_wrapper.py`: High-accuracy segmentation wrapper.
- `models/camouflage_yolo11_best.pt`: Best trained weights.
- `training_results/`: Detailed PR-curves, confusion matrices, and mAP plots.

---
### 5. Subject Context: SPML
This project demonstrates that **Signal Processing** is not just a preprocessing step but a robust feature extraction methodology. By understanding **Spectral Entropy** (via FFT) and **Multi-Scale Detail Energy** (via Wavelets), we can build machine learning models that are more interpretable and efficient than "black-box" neural networks alone.

*Developed as a project for Signal Processing for Machine Learning (SPML).*
