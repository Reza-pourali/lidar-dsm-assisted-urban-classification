"""Generate TIN/IDW DSM rasters from a LAS/LAZ point cloud."""

from pathlib import Path
import argparse
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidar_dsm_classification.dsm import generate_dsm_products, raster_min_max


def parse_args():
    p = argparse.ArgumentParser(description="Generate DSMs with WhiteboxTools.")
    p.add_argument("las_file")
    p.add_argument("--output-dir", default="dsm_outputs")
    p.add_argument("--resolution", type=float, default=1.5)
    p.add_argument("--idw-weight", type=float, default=2.0)
    p.add_argument("--idw-radius", type=float, default=2.5)
    return p.parse_args()


def main():
    args = parse_args()
    outputs = generate_dsm_products(
        args.las_file,
        args.output_dir,
        resolution=args.resolution,
        idw_weight=args.idw_weight,
        idw_radius=args.idw_radius,
    )

    rows = []
    for output in outputs:
        mn, mx = raster_min_max(output)
        rows.append((output.name, mn, mx))
        print(f"{output.name}: min={mn:.3f} m, max={mx:.3f} m")

    csv_path = Path(args.output_dir) / "dsm_ranges.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raster", "min_elevation_m", "max_elevation_m"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
