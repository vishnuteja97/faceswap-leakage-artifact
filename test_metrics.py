"""Boundary cases for the shared scientific metrics; no external test framework."""
import unittest
import numpy as np
from metrics import auc, operating_points, quantile_tpr


class MetricChecks(unittest.TestCase):
    def test_auc_ties_and_orientation(self):
        self.assertEqual(auc([1, 2], [0, 1]), .875)
        self.assertEqual(auc([1, 1], [1, 1]), .5)
        self.assertEqual(auc([0, 1], [2, 3]), 0)

    def test_auc_against_pairwise_definition(self):
        rng = np.random.default_rng(7)
        p, n = rng.integers(0, 5, 37), rng.integers(0, 5, 29)
        expected = np.mean((p[:, None] > n) + .5 * (p[:, None] == n))
        self.assertAlmostEqual(auc(p, n), expected)

    def test_small_negative_class_cannot_allow_one_false_positive(self):
        p, n = np.array([.9, 1.1]), np.array([0., 1.])
        self.assertEqual(operating_points(p, n)['tpr_1pct'], .5)
        self.assertEqual(operating_points([0], [1])['tpr_1pct'], 0)

    def test_quantile_fpr_is_not_a_strict_budget(self):
        p, n = np.array([.999, 2.]), np.array([0., 1.])
        self.assertEqual(quantile_tpr(p, n, .01), (1., .5))
        self.assertEqual(operating_points(p, n)['tpr_1pct'], .5)

    def test_reject_invalid_scores(self):
        with self.assertRaises(ValueError):
            auc([np.nan], [0])
        with self.assertRaises(ValueError):
            operating_points([], [0])


if __name__ == '__main__':
    unittest.main()
