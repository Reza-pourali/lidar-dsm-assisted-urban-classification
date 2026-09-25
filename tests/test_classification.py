import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidar_dsm_classification.classification import run_experiment
from lidar_dsm_classification.sampling import stratified_pixel_split


class TestClassification(unittest.TestCase):
    def make_scene(self):
        rng = np.random.default_rng(7)
        rows, cols = 50, 50
        roi = np.zeros((rows, cols), dtype=np.uint8)

        # Five compact class regions.
        roi[2:12, 2:22] = 1
        roi[14:24, 2:22] = 2
        roi[26:36, 2:22] = 3
        roi[2:17, 28:48] = 4
        roi[20:35, 28:48] = 5

        labels = roi.reshape(-1)
        n = rows * cols

        # Seven-band spectral/texture stack.
        base = np.zeros((n, 7), dtype=np.float32)
        for cls in range(1, 6):
            idx = np.flatnonzero(labels == cls)
            center = cls * 2.0
            base[idx] = rng.normal(center, 0.15, size=(len(idx), 7))

        # Background receives arbitrary values but is never sampled.
        bg = np.flatnonzero(labels == 0)
        base[bg] = rng.normal(0, 1, size=(len(bg), 7))

        without = base.reshape(rows, cols, 7)

        # Add one elevation band that is strongly class-dependent.
        dsm = np.zeros(n, dtype=np.float32)
        for cls in range(1, 6):
            idx = np.flatnonzero(labels == cls)
            dsm[idx] = cls * 10.0 + rng.normal(0, 0.2, size=len(idx))
        dsm[bg] = rng.normal(0, 1, size=len(bg))

        with_dsm = np.concatenate(
            [without, dsm.reshape(rows, cols, 1)],
            axis=2,
        )
        return without, with_dsm, roi

    def test_experiment_runs_and_predictions_match_scene_shape(self):
        without, with_dsm, roi = self.make_scene()
        train_idx, test_idx, labels = stratified_pixel_split(
            roi, test_ratio=0.30, random_state=42
        )

        result = run_experiment(
            without,
            with_dsm,
            labels,
            train_idx,
            test_idx,
            random_state=42,
            rf_estimators=30,
        )

        for group in [result.without_dsm, result.with_dsm]:
            for model_result in group.values():
                self.assertEqual(model_result.prediction.shape, roi.shape)
                self.assertGreater(model_result.accuracy, 0.95)
                self.assertEqual(model_result.confusion_matrix.shape, (5, 5))

    def test_with_dsm_must_have_one_extra_band(self):
        without, with_dsm, roi = self.make_scene()
        train_idx, test_idx, labels = stratified_pixel_split(roi)

        with self.assertRaises(ValueError):
            run_experiment(
                without,
                without,
                labels,
                train_idx,
                test_idx,
            )


if __name__ == "__main__":
    unittest.main()
