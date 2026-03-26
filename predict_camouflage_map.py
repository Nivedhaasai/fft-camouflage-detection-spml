from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.config import PATCH_SIZE, STRIDE, CAMO_PROB_THRESHOLD
from src.fft_features import extract_multiscale_features, get_multiscale_feature_names


def predict_camouflage(
    image_path: str,
    model_path: str = "models/svm_fft_camouflage_model.joblib",
    output_path: str = "outputs/prediction.png",
):
    """Perform camouflage detection using multi-scale features and academic visualization."""
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found: {model_path}. Please train it first.")
        return

    pipeline = joblib.load(model_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    
    # Feature Consistency Validation
    expected_len = len(get_multiscale_feature_names())
    # One-time feature extraction to check length
    test_feature = extract_multiscale_features(img_gray, 0, 0)
    if len(test_feature) != expected_len:
         print(f"[CRITICAL ERROR] Feature length mismatch! Training={expected_len}, Extraction={len(test_feature)}")
         return

    # Sliding window extraction
    patches_features = []
    coords = []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patches_features.append(extract_multiscale_features(img_gray, y, x))
            coords.append((y, x))
            
    # Batch predict
    X = np.array(patches_features, dtype=np.float32)
    
    # Check if SVM was trained with probability support
    if hasattr(pipeline.named_steps["svm"], "predict_proba") and pipeline.named_steps["svm"].probability:
        # Step 8 requested probability=False, but we handle both cases for robustness.
        # If no probability, we use decison_function or binary labels.
        probs = pipeline.predict_proba(X)[:, 1]
    else:
        # Use decision_function for probability-like values if probability=False
        scores = pipeline.decision_function(X)
        # Convert to 0..1 via sigmoid for heatmap
        probs = 1 / (1 + np.exp(-scores))
        
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    for (y, x), p in zip(coords, probs):
        prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += p
        count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0
        
    prob_map = prob_map / (count_map + 1e-8)
    
    # Step 11: Gaussian Smoothing
    prob_map_smoothed = cv2.GaussianBlur(prob_map, (5, 5), sigmaX=1.5)
    
    # Masking for bounding boxes
    mask = (prob_map_smoothed > CAMO_PROB_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img_bgr.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
    # FFT magnitude for visualization (center of image)
    h_c, w_c = H // 2, W // 2
    vis_patch = img_gray[h_c-32:h_c+32, w_c-32:w_c+32]
    f_shift = np.fft.fftshift(np.fft.fft2(vis_patch))
    mag_spec = np.log(np.abs(f_shift) + 1)
    
    # Step 13: Visualization Panel (4 panels)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[1].imshow(mag_spec, cmap="magma")
    axes[1].set_title("FFT Magnitude (Center)")
    im2 = axes[2].imshow(prob_map_smoothed, cmap="viridis")
    axes[2].set_title("Probability Heatmap")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Detections")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Prediction saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/svm_fft_camouflage_model.joblib")
    parser.add_argument("--output", default="outputs/prediction_v2.png")
    args = parser.parse_args()
    predict_camouflage(args.image, args.model, args.output)
