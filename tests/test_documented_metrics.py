import csv
import sys
import unittest
from pathlib import Path


class TestDocumentedMetrics(unittest.TestCase):
    def test_reported_accuracy_improvement(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "documented_classification_metrics.csv"

        rows = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[(row["model"], row["feature_set"])] = float(
                    row["overall_accuracy_percent"]
                )

        self.assertAlmostEqual(
            rows[("Decision Tree", "with DSM")]
            - rows[("Decision Tree", "without DSM")],
            5.78,
            places=2,
        )
        self.assertAlmostEqual(
            rows[("Random Forest", "with DSM")]
            - rows[("Random Forest", "without DSM")],
            3.88,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
