"""Raster input-output utilities based on rasterio."""

from pathlib import Path
import numpy as np


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Raster I/O requires rasterio. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return rasterio


def read_feature_stack(path):
    """Read a raster stack as rows x cols x bands."""
    rasterio = _rasterio()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()

    if data.ndim != 3:
        raise ValueError("Feature raster must contain one or more bands.")

    return np.moveaxis(data, 0, -1), profile


def read_label_raster(path):
    """Read the first band of an ROI/label raster."""
    rasterio = _rasterio()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        labels = src.read(1)
        profile = src.profile.copy()

    return labels, profile


def write_classification_raster(path, labels_2d, reference_profile):
    """Write a single-band uint8 classification GeoTIFF."""
    rasterio = _rasterio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = reference_profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(labels_2d, dtype=np.uint8), 1)
