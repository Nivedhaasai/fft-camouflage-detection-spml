# FFT-Based Camouflage Detection (SPML Academic Prototype)

Overview
This project provides an academic-quality signal-processing and classical machine learning pipeline for detecting camouflaged regions using **Multi-Scale FFT** and **Gabor Texture Features**. It avoids deep learning, focusing on frequency-domain analysis and texture-based discrimination.

---

## ?? Key Improvements (v2.0)
- **Multi-Scale Feature Fusion**: Concatenates features from 32x32, 64x64, and 96x96 patch scales.
- **Enhanced Feature Vector (81-Dim)**:
  - **FFT Features**: Mean/Std energy, Shannon entropy, High/Mid frequency ratios, 16 Radial profile bins, and Directional energy.
  - **Gabor Features**: Mean energy from 4 orientations (0°, 45°, 90°, 135°).
- **Deterministic Pipeline**: Unified configuration in `src/config.py` with `RANDOM_SEED=42`.
- **Balanced Subsampling**: Training is limited to 40,000 patches (20k Camo / 20k Background) for optimal speed and class balance.
- **Academic Visualization**: Provides a 4-panel output including Original Image, FFT Spectrum, Probability Heatmap, and Bounding Box Detections.

---

## ?? Dataset Structure
```text
dataset/camo/       # Camouflage images
dataset/non_camo/   # Background-only images
masks/              # Binary masks for camo images (matching filenames)
```
- **Verification**: The training script automatically verifies that every camo image has a matching mask in the `masks/` folder.

---

## ??? Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ??? Training
To retrain the SVM classifier with the upgraded feature pipeline:
```bash
python train_fft_camouflage.py
```
- **Output**: Generates `models/svm_fft_camouflage_model.joblib`.
- **Metrics**: Prints Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## ?? Inference & Visualization
Run the detection on a test image:
```bash
python predict_camouflage_map.py --image "dataset/camo/camourflage_00001.jpg" --output "outputs/prediction_v2.png"
```
The output visualization includes:
1. **Original Image**
2. **FFT Magnitude Spectrum** (Center patch)
3. **Probability Heatmap** (Gaussian-smoothed)
4. **Camouflage Overlay** (Bounding boxes filtered by `CAMO_PROB_THRESHOLD`)

---

## ?? Configuration
Modify `src/config.py` to tune the system:
- `PATCH_SIZE`: Base patch scale.
- `STRIDE`: Overlap for sliding window.
- `CAMO_PROB_THRESHOLD`: Sensitivity for bounding box generation.
