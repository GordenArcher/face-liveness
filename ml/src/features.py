"""Local Binary Pattern feature extraction. This is the whole idea
behind the classical baseline: real skin has micro-texture a printed
photo or a screen replay doesn't reproduce at the same frequency, LBP
captures that directly as a histogram of local texture patterns rather
than needing a learned feature extractor to discover it.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Fixed size so every image produces a histogram of comparable scale,
# NUAA images vary in resolution, resizing first is what makes LBP
# histograms from different source images actually comparable.
IMAGE_SIZE = (128, 128)

# 8 neighbors at radius 1 is the standard baseline LBP configuration in
# the face anti-spoofing literature, not tuned further here, that's
# exactly the point of this being step one, not the final model.
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"

# Uniform LBP with P=8 has P+2 = 10 distinct pattern bins, hardcoded
# here rather than derived, so a typo in LBP_POINTS above would show up
# as a shape mismatch instead of a silently wrong histogram.
N_BINS = LBP_POINTS + 2


def extract_lbp_histogram(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read image at {image_path}, is it corrupted or an unsupported format?")

    image = cv2.resize(image, IMAGE_SIZE)
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)

    histogram, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, N_BINS + 1), range=(0, N_BINS)
    )
    # Normalized so histogram scale doesn't just reflect image size,
    # without this an SVM would partly be learning "how many pixels"
    # instead of "what texture pattern".
    histogram = histogram.astype(np.float64)
    histogram /= histogram.sum() + 1e-7
    return histogram


def extract_features(samples) -> np.ndarray:
    return np.array([extract_lbp_histogram(s.path) for s in samples])