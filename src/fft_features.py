from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from src.config import (
    RADIAL_BINS, SCALES, GABOR_ORIENTATIONS, 
    LOG_POLAR_RADIAL_BANDS, LOG_POLAR_ANGULAR_SECTORS,
    LBP_RADIUS, LBP_POINTS
)

def get_fft_feature_names(radial_bins: int = RADIAL_BINS) -> list[str]:
    radial_names = [f"radial_spectrum_energy_bin_{i:02d}" for i in range(radial_bins)]
    return [
        "mean_spectral_energy",
        "std_spectral_energy",
        "shannon_entropy",
        "high_frequency_energy_ratio",
        "mid_frequency_energy_ratio",
        *radial_names,
        "horizontal_directional_energy",
        "vertical_directional_energy",
    ]

def get_gabor_feature_names() -> list[str]:
    return [f"gabor_energy_{theta}" for theta in GABOR_ORIENTATIONS]

def get_log_polar_feature_names() -> list[str]:
    names = []
    for r in range(LOG_POLAR_RADIAL_BANDS):
        for t in range(LOG_POLAR_ANGULAR_SECTORS):
            names.append(f"log_polar_r{r}_t{t}")
    return names

def get_lbp_feature_names() -> list[str]:
    # Uniform LBP for 8 points has P+2 bins
    return [f"lbp_bin_{i:02d}" for i in range(LBP_POINTS + 2)]

def get_multiscale_feature_names() -> list[str]:
    all_names = []
    fft_names = get_fft_feature_names()
    gabor_names = get_gabor_feature_names()
    lp_names = get_log_polar_feature_names()
    lbp_names = get_lbp_feature_names()
    
    base_names = fft_names + gabor_names + lp_names + lbp_names
    for scale in SCALES:
        all_names.extend([f"s{scale}_{name}" for name in base_names])
    return all_names

def radial_profile(power_spectrum: np.ndarray, bins: int = 16) -> np.ndarray:
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)
    features = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (r_norm >= lo) & (r_norm < hi)
        features.append(float(power_spectrum[mask].mean()) if np.any(mask) else 0.0)
    return np.asarray(features, dtype=np.float32)

def shannon_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.min()
    total = values.sum()
    if total <= 0: return 0.0
    p = values / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def extract_log_polar_features(magnitude: np.ndarray) -> np.ndarray:
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    
    # Simple sectors and bands integration
    y, x = np.indices((h, w))
    dy, dx = y - center[0], x - center[1]
    
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) # -pi to pi
    
    # Radial bands (log-scaled)
    # Avoid log(0)
    epsilon = 1e-5
    log_dist = np.log(dist + epsilon)
    log_max = np.log(max_radius + epsilon)
    log_min = np.log(epsilon)
    
    r_bins = np.linspace(log_min, log_max, LOG_POLAR_RADIAL_BANDS + 1)
    t_bins = np.linspace(-np.pi, np.pi, LOG_POLAR_ANGULAR_SECTORS + 1)
    
    features = []
    for i in range(LOG_POLAR_RADIAL_BANDS):
        for j in range(LOG_POLAR_ANGULAR_SECTORS):
            mask = (log_dist >= r_bins[i]) & (log_dist < r_bins[i+1]) & \
                   (angle >= t_bins[j]) & (angle < t_bins[j+1])
            features.append(float(magnitude[mask].mean()) if np.any(mask) else 0.0)
    
    return np.asarray(features, dtype=np.float32)

def extract_single_patch_features(patch: np.ndarray) -> np.ndarray:
    # 1. FFT Features
    patch_fft = patch.astype(np.float32)
    patch_fft = patch_fft - patch_fft.mean()
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1])).astype(np.float32)
    patch_windowed = patch_fft * window
    fft_map = np.fft.fftshift(np.fft.fft2(patch_windowed))
    magnitude = np.abs(fft_map)
    power = magnitude ** 2
    total_energy = float(power.sum()) + 1e-8
    
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r_norm = np.sqrt((x-cx)**2 + (y-cy)**2) / (np.sqrt(cx**2+cy**2) + 1e-8)
    
    high_freq_ratio = float(power[r_norm >= 0.5].sum() / total_energy)
    mid_freq_ratio = float(power[(r_norm >= 0.2) & (r_norm < 0.5)].sum() / total_energy)
    radial = radial_profile(power, bins=RADIAL_BINS)
    radial = radial / (radial.sum() + 1e-8)
    
    horiz_energy = float(power[np.abs(x-cx) < 2].sum() / total_energy)
    vert_energy = float(power[np.abs(y-cy) < 2].sum() / total_energy)
    
    fft_vec = np.array([float(power.mean()), float(power.std()), shannon_entropy(power),
                        high_freq_ratio, mid_freq_ratio, *radial, horiz_energy, vert_energy], dtype=np.float32)
    
    # 2. Gabor Features
    gabor_vec = []
    for theta_deg in GABOR_ORIENTATIONS:
        kernel = cv2.getGaborKernel((31, 31), 4.0, theta_deg*np.pi/180.0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(patch.astype(np.float32), cv2.CV_32F, kernel)
        gabor_vec.append(float(np.mean(filtered**2)))
    
    # 3. Log-Polar Spectral Pooling (Step 1)
    lp_vec = extract_log_polar_features(magnitude)
    
    # 4. LBP (Step 2)
    lbp = local_binary_pattern(patch, LBP_POINTS, LBP_RADIUS, method="uniform")
    # Histogram of uniform LBP has P+2 bins
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return np.concatenate([fft_vec, np.array(gabor_vec, dtype=np.float32), lp_vec, hist.astype(np.float32)])

def extract_multiscale_features(image: np.ndarray, y: int, x: int, base_patch_size: int = 64) -> np.ndarray:
    fused_features = []
    H, W = image.shape
    cy, cx = y + base_patch_size // 2, x + base_patch_size // 2
    for scale in SCALES:
        half = scale // 2
        y1, y2, x1, x2 = cy - half, cy + half, cx - half, cx + half
        if y1 < 0 or y2 > H or x1 < 0 or x2 > W:
            p_y1, p_y2, p_x1, p_x2 = max(0, y1), min(H, y2), max(0, x1), min(W, x2)
            crop = image[p_y1:p_y2, p_x1:p_x2]
            patch = np.pad(crop, ((p_y1-y1, y2-p_y2), (p_x1-x1, x2-p_x2)), mode="edge")
        else:
            patch = image[y1:y2, x1:x2]
        fused_features.append(extract_single_patch_features(patch))
    return np.concatenate(fused_features)
