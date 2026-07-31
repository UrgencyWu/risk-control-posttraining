import unittest

import numpy as np

from src.evaluation.metrics import apply_threshold, select_cost_threshold


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_lowest_cost_threshold_from_validation_scores(self):
        scores = np.array([0.10, 0.20, 0.80, 0.90])
        labels = np.array([0, 0, 1, 1])

        threshold, cost = select_cost_threshold(scores, labels, threshold_grid=(0.15, 0.50, 0.85))

        self.assertEqual(0.50, threshold)
        self.assertEqual(0, cost)

    def test_ties_keep_the_lowest_threshold_deterministically(self):
        threshold, cost = select_cost_threshold(
            scores=[0.9, 0.8], ground_truth=[1, 1], threshold_grid=(0.10, 0.20, 0.30)
        )

        self.assertEqual(0.10, threshold)
        self.assertEqual(0, cost)

    def test_threshold_application_rejects_invalid_scores(self):
        with self.assertRaises(ValueError):
            apply_threshold([0.2, float("nan")], 0.5)


if __name__ == "__main__":
    unittest.main()
