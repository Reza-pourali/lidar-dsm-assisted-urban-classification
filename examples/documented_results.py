"""Print the documented final experiment results from the coursework."""

from pathlib import Path
import csv


def main():
    path = Path(__file__).resolve().parents[1] / "data" / "documented_classification_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(
                f"{row['model']} | {row['feature_set']} | "
                f"accuracy={float(row['overall_accuracy_percent']):.2f}% | "
                f"macro-F1={float(row['macro_f1']):.2f} | "
                f"weighted-F1={float(row['weighted_f1']):.2f}"
            )


if __name__ == "__main__":
    main()
