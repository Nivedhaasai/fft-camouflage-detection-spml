from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.config import PATCH_SIZE, STRIDE, MIN_REGION_AREA
from src.fft_features import extract_multiscale_features, get_multiscale_feature_names

def predict_camouflage(
    image_path: str,
    model_path: str = "models/svm_fft_camouflage_model.joblib",
    output_path: str = "outputs/prediction.png",
):
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found: {model_path}.")
        return

    # Load dict with model and optimized threshold
    data = joblib.load(model_path)
    pipeline = data["model"]
    # Step 7: Use automatically optimized threshold
    camo_prob_threshold = data["threshold"]

    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    
    expected_len = len(get_multiscale_feature_names())
    
    patches_features = []
    coords = []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patches_features.append(extract_multiscale_features(img_gray, y, x))
            coords.append((y, x))
            
    X = np.array(patches_features, dtype=np.float32)
    # Step 4: Calibrated probability estimation
    probs = pipeline.predict_proba(X)[:, 1]
        
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    for (y, x), p in zip(coords, probs):
        prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += p
        count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0
    prob_map /= (count_map + 1e-8)
    
    prob_map_smoothed = cv2.GaussianBlur(prob_map, (5, 5), sigmaX=1.5)
    
    # Step 3: Connected Component Filtering
    mask = (prob_map_smoothed > camo_prob_threshold).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Filter by area
    refined_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_REGION_AREA:
            refined_mask[labels == i] = 255
            
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_bgr.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    
    h_c, w_c = H // 2, W // 2
    vis_patch = img_gray[max(0, h_c-32):min(H, h_c+32), max(0, w_c-32):min(W, w_c+32)]
    if vis_patch.size > 0:
        f_shift = np.fft.fftshift(np.fft.fft2(vis_patch))
        axes[1].imshow(np.log(np.abs(f_shift) + 1), cmap="magma")
    axes[1].set_title("FFT Magnitude (Center)")
    
    im2 = axes[2].imshow(prob_map_smoothed, cmap="viridis")
    axes[2].set_title(f"Heatmap (T={camo_prob_threshold:.2f})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Refined Areas (>{MIN_REGION_AREA}px)")
    
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Prediction saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/svm_fft_camouflage_model.joblib")
    parser.add_argument("--output", default="outputs/prediction_v3.png")
    args = parser.parse_args()
    predict_camouflage(args.image, args.model, args.output)
