"""Generate DSM rasters from LiDAR using WhiteboxTools.

The original script cast elevation arrays to uint8 before visualization.
That destroys the physical elevation scale. This refactor preserves the
floating-point DSM values and plots them directly.
"""

from pathlib import Path


RETURN_TYPES = ("first", "last", "all")


def _whitebox():
    try:
        import whitebox
    except ImportError as exc:
        raise ImportError(
            "DSM generation requires the `whitebox` package. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return whitebox


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "DSM visualization requires rasterio."
        ) from exc
    return rasterio


def generate_dsm_products(
    las_file,
    output_dir,
    resolution=1.5,
    idw_weight=2.0,
    idw_radius=2.5,
):
    """Generate TIN and IDW DSMs for first, last, and all returns."""
    whitebox = _whitebox()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las_file = Path(las_file)
    if not las_file.exists():
        raise FileNotFoundError(las_file)

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(output_dir))

    outputs = []
    for return_type in RETURN_TYPES:
        tin_output = output_dir / f"dsm_tin_{return_type}.tif"
        wbt.lidar_tin_gridding(
            i=str(las_file),
            output=str(tin_output),
            parameter="elevation",
            returns=return_type,
            resolution=resolution,
        )
        outputs.append(tin_output)

        idw_output = output_dir / f"dsm_idw_{return_type}.tif"
        wbt.lidar_idw_interpolation(
            i=str(las_file),
            output=str(idw_output),
            parameter="elevation",
            returns=return_type,
            resolution=resolution,
            weight=idw_weight,
            radius=idw_radius,
        )
        outputs.append(idw_output)

    return outputs


def raster_min_max(path):
    """Return finite/nodata-aware min and max elevation for a DSM raster."""
    rasterio = _rasterio()
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
    return float(data.min()), float(data.max())
