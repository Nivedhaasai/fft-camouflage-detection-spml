from __future__ import annotations

import cv2
import numpy as np
from src.config import RADIAL_BINS, SCALES, GABOR_ORIENTATIONS


def get_fft_feature_names(radial_bins: int = RADIAL_BINS) -> list[str]:
    """Return deterministic FFT feature names in the exact extraction order."""
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
    """Return Gabor feature names for defined orientations."""
    return [f"gabor_energy_{theta}" for theta in GABOR_ORIENTATIONS]


def get_multiscale_feature_names() -> list[str]:
    """Return all feature names concatenated across scales."""
    all_names = []
    fft_names = get_fft_feature_names()
    gabor_names = get_gabor_feature_names()
    base_names = fft_names + gabor_names
    for scale in SCALES:
        all_names.extend([f"s{scale}_{name}" for name in base_names])
    return all_names


def radial_profile(power_spectrum: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compute a binned radial average of a 2D power spectrum."""
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)

    features = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (r_norm >= lo) & (r_norm < hi)
        if np.any(mask):
            features.append(float(power_spectrum[mask].mean()))
        else:
            features.append(0.0)

    return np.asarray(features, dtype=np.float32)


def shannon_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.min()
    total = values.sum()
    if total <= 0:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def extract_single_patch_features(patch: np.ndarray) -> np.ndarray:
    """Extract FFT and Gabor features for a single patch (agnostic of its scale)."""
    # 1. FFT Features
    patch_fft = patch.astype(np.float32)
    patch_fft = patch_fft - patch_fft.mean()
    wy = np.hanning(patch_fft.shape[0])
    wx = np.hanning(patch_fft.shape[1])
    window = np.outer(wy, wx).astype(np.float32)
    patch_windowed = patch_fft * window

    fft_map = np.fft.fftshift(np.fft.fft2(patch_windowed))
    magnitude = np.abs(fft_map)
    power = magnitude ** 2
    total_energy = float(power.sum()) + 1e-8

    mean_spectral_energy = float(power.mean())
    std_spectral_energy = float(power.std())
    spectral_entropy = shannon_entropy(power)

    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)

    high_freq_mask = r_norm >= 0.5
    high_freq_ratio = float(power[high_freq_mask].sum() / total_energy)

    mid_freq_mask = (r_norm >= 0.2) & (r_norm < 0.5)
    mid_freq_ratio = float(power[mid_freq_mask].sum() / total_energy)

    radial = radial_profile(power, bins=RADIAL_BINS)
    radial = radial / (radial.sum() + 1e-8)

    # Directional energy (Horizontal vs Vertical)
    horiz_mask = np.abs(x - cx) < 2
    horiz_energy = float(power[horiz_mask].sum() / total_energy)
    vert_mask = np.abs(y - cy) < 2
    vert_energy = float(power[vert_mask].sum() / total_energy)

    fft_vec = np.array([
        mean_spectral_energy,
        std_spectral_energy,
        spectral_entropy,
        high_freq_ratio,
        mid_freq_ratio,
        *radial,
        horiz_energy,
        vert_energy
    ], dtype=np.float32)

    # 2. Gabor Features
    gabor_vec = []
    ksize = 31
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    for theta_deg in GABOR_ORIENTATIONS:
        theta = theta_deg * np.pi / 180.0
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(patch.astype(np.float32), cv2.CV_32F, kernel)
        gabor_vec.append(float(np.mean(filtered**2)))
    
    return np.concatenate([fft_vec, np.array(gabor_vec, dtype=np.float32)])


def extract_multiscale_features(image: np.ndarray, y: int, x: int, base_patch_size: int = 64) -> np.ndarray:
    """Extract and fuse features from multiple scales centered at (y,x)."""
    fused_features = []
    H, W = image.shape
    cy, cx = y + base_patch_size // 2, x + base_patch_size // 2

    for scale in SCALES:
        half = scale // 2
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        
        if y1 < 0 or y2 > H or x1 < 0 or x2 > W:
            p_y1, p_y2 = max(0, y1), min(H, y2)
            p_x1, p_x2 = max(0, x1), min(W, x2)
            crop = image[p_y1:p_y2, p_x1:p_x2]
            
            pad_y1 = p_y1 - y1
            pad_y2 = y2 - p_y2
            pad_x1 = p_x1 - x1
            pad_x2 = x2 - p_x2
            patch = np.pad(crop, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode="edge")
        else:
            patch = image[y1:y2, x1:x2]
            
        fused_features.append(extract_single_patch_features(patch))
        
    return np.concatenate(fused_features)
