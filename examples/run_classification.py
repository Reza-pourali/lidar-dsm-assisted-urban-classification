"""Run the DSM-vs-no-DSM urban classification experiment."""

from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidar_dsm_classification.classification import run_experiment
from lidar_dsm_classification.raster_io import (
    read_feature_stack,
    read_label_raster,
    write_classification_raster,
)
from lidar_dsm_classification.sampling import stratified_pixel_split
from lidar_dsm_classification.visualization import (
    save_classification_map,
    save_confusion_matrix,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Decision Tree and Random Forest with and without DSM."
    )
    p.add_argument("--without-dsm", required=True, help="Feature stack without DSM")
    p.add_argument("--with-dsm", required=True, help="Feature stack with DSM")
    p.add_argument("--roi", required=True, help="ROI label raster (0=background, 1..5=classes)")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--test-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rf-estimators", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    without_dsm, without_profile = read_feature_stack(args.without_dsm)
    with_dsm, with_profile = read_feature_stack(args.with_dsm)
    roi, _ = read_label_raster(args.roi)

    if roi.shape != without_dsm.shape[:2] or roi.shape != with_dsm.shape[:2]:
        raise ValueError("ROI and feature rasters must have matching rows/columns.")

    train_idx, test_idx, labels = stratified_pixel_split(
        roi,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    experiment = run_experiment(
        without_dsm,
        with_dsm,
        labels,
        train_idx,
        test_idx,
        random_state=args.seed,
        rf_estimators=args.rf_estimators,
    )

    metrics = []
    for group_name, result_group, profile in [
        ("without_dsm", experiment.without_dsm, without_profile),
        ("with_dsm", experiment.with_dsm, with_profile),
    ]:
        for model_name, result in result_group.items():
            slug = model_name.lower().replace(" ", "_")
            prefix = f"{slug}_{group_name}"

            write_classification_raster(
                output_dir / f"{prefix}.tif",
                result.prediction,
                profile,
            )
            save_classification_map(
                result.prediction,
                output_dir / f"{prefix}.png",
                f"{model_name} - {result.feature_set}",
            )
            save_confusion_matrix(
                result.confusion_matrix,
                output_dir / f"{prefix}_confusion_matrix.png",
                f"{model_name} - {result.feature_set}",
            )

            metrics.append({
                "model": model_name,
                "feature_set": result.feature_set,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
            })

            if result.feature_importances is not None:
                with (output_dir / f"{prefix}_feature_importance.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as f:
                    w = csv.writer(f)
                    w.writerow(["band", "importance"])
                    for i, value in enumerate(result.feature_importances, start=1):
                        w.writerow([i, float(value)])

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training pixels: {len(train_idx):,}")
    print(f"Test pixels: {len(test_idx):,}")
    for row in metrics:
        print(
            f"{row['model']} | {row['feature_set']} | "
            f"accuracy={100 * row['accuracy']:.2f}% | "
            f"macro-F1={row['macro_f1']:.3f} | "
            f"weighted-F1={row['weighted_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
