import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidar_dsm_classification.sampling import stratified_pixel_split


class TestSampling(unittest.TestCase):
    def test_background_is_excluded_and_split_is_reproducible(self):
        roi = np.array([
            [0, 1, 1, 1, 2, 2, 2],
            [0, 1, 1, 1, 2, 2, 2],
            [0, 3, 3, 3, 4, 4, 4],
            [0, 3, 3, 3, 4, 4, 4],
            [0, 5, 5, 5, 5, 5, 5],
        ])

        a_train, a_test, labels = stratified_pixel_split(
            roi, test_ratio=0.33, random_state=42
        )
        b_train, b_test, _ = stratified_pixel_split(
            roi, test_ratio=0.33, random_state=42
        )

        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_test, b_test)
        self.assertTrue(np.all(labels[a_train] != 0))
        self.assertTrue(np.all(labels[a_test] != 0))

        self.assertEqual(set(np.unique(labels[a_train])), {1, 2, 3, 4, 5})
        self.assertEqual(set(np.unique(labels[a_test])), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
