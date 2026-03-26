from __future__ import annotations

# Shared deterministic configuration across data loading, training, and prediction.
PATCH_SIZE = 64
STRIDE = 32
MASK_THRESHOLD = 0.1
RANDOM_SEED = 42
CAMO_PROB_THRESHOLD = 0.6

# FFT feature schema configuration.
RADIAL_BINS = 16
SCALES = [32, 64, 96]
GABOR_ORIENTATIONS = [0, 45, 90, 135]

