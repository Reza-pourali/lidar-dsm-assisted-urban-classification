"""Reproducible train/test sampling from labeled ROI pixels."""

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_pixel_split(
    roi,
    test_ratio=0.30,
    random_state=42,
    background_label=0,
):
    """Split labeled ROI pixels into reproducible stratified train/test sets.

    Background pixels are excluded from both train and test.
    """
    roi = np.asarray(roi)
    if roi.ndim != 2:
        raise ValueError("ROI raster must be two-dimensional.")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1.")

    labels = roi.reshape(-1)
    labeled_idx = np.flatnonzero(labels != background_label)

    if labeled_idx.size == 0:
        raise ValueError("ROI contains no labeled pixels.")

    labeled_y = labels[labeled_idx]
    classes, counts = np.unique(labeled_y, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("Each non-background class needs at least two labeled pixels.")

    train_idx, test_idx = train_test_split(
        labeled_idx,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labeled_y,
    )

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
        labels,
    )
