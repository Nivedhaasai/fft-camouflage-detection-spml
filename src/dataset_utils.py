from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import random
from tqdm import tqdm

from src.config import MASK_THRESHOLD, PATCH_SIZE, STRIDE, RANDOM_SEED
from src.fft_features import extract_multiscale_features


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def find_mask_for_image(image_path: Path, masks_dir: Path) -> Path | None:
    # Try different extensions for the mask
    for ext in [image_path.suffix, ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = masks_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def list_camo_image_mask_pairs(camo_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """Verify and list camo image-mask pairs."""
    pairs = []
    
    # Check camo_dir and also dataset_acd/images
    search_dirs = [
        (camo_dir, masks_dir),
        (camo_dir.parent.parent / "dataset_acd" / "images", camo_dir.parent.parent / "dataset_acd" / "masks")
    ]
    
    for img_dir, msk_dir in search_dirs:
        if not img_dir.exists() or not msk_dir.exists():
            continue
            
        images = list_images(img_dir)
        for img_path in images:
            msk_path = find_mask_for_image(img_path, msk_dir)
            if msk_path:
                pairs.append((img_path, msk_path))
            else:
                print(f"[WARN] Missing mask for: {img_path.name}")
                
    return pairs


def build_multiscale_patch_dataset(
    camo_dir: Path,
    non_camo_dir: Path,
    masks_dir: Path,
    limit: int = 40000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build multi-scale feature dataset with balanced sampling and verification."""
    random.seed(RANDOM_SEED)
    
    camo_pairs = list_camo_image_mask_pairs(camo_dir, masks_dir)
    non_camo_imgs = list_images(non_camo_dir)
    
    print(f"Valid camo-mask pairs: {len(camo_pairs)}")
    print(f"Non-camo (background) images: {len(non_camo_imgs)}")
    
    features_camo = []
    features_bg = []
    
    # Extract from camo images
    for img_path, msk_path in tqdm(camo_pairs, desc="Extracting camo patches"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        if img.shape != msk.shape:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        H, W = img.shape
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                p_msk = msk[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                mask_mean = float(p_msk.mean() / 255.0)
                feat = extract_multiscale_features(img, y, x)
                if mask_mean > MASK_THRESHOLD:
                    features_camo.append(feat)
                else:
                    features_bg.append(feat)
                    
    # Extract from background-only images
    for img_path in tqdm(non_camo_imgs, desc="Extracting background patches"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                features_bg.append(extract_multiscale_features(img, y, x))

    print(f"Total patches - Camo: {len(features_camo)}, Background: {len(features_bg)}")
    
    # Balanced subsampling
    half_limit = limit // 2
    if len(features_camo) > half_limit:
        features_camo = random.sample(features_camo, half_limit)
    if len(features_bg) > half_limit:
        features_bg = random.sample(features_bg, half_limit)
        
    X = np.array(features_camo + features_bg, dtype=np.float32)
    y = np.array([1]*len(features_camo) + [0]*len(features_bg), dtype=np.int32)
    
    # Shuffle
    idx = np.random.choice(len(X), len(X), replace=False)
    return X[idx], y[idx]
