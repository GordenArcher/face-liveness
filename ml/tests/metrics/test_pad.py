"""APCER/BPCER/ACER must follow the presentation-attack definitions.

These are the headline numbers the whole project reports, so a silent
swap between APCER and BPCER would misreport the security risk.
"""

import unittest

import numpy as np

from liveness.metrics.pad import compute_metrics


class ComputeMetricsTest(unittest.TestCase):
    def test_apcer_counts_spoofs_accepted_as_live(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = compute_metrics(y_true, y_pred)

        # One of two spoofs accepted, one of two lives rejected.
        self.assertAlmostEqual(metrics["apcer"], 0.5)
        self.assertAlmostEqual(metrics["bpcer"], 0.5)
        self.assertAlmostEqual(metrics["acer"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)

    def test_all_spoofs_accepted_is_maximum_apcer(self):
        y_true = np.array([1, 0, 0])
        y_pred = np.array([1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["apcer"], 1.0)
        self.assertAlmostEqual(metrics["bpcer"], 0.0)
        self.assertAlmostEqual(metrics["acer"], 0.5)

    def test_missing_class_yields_nan_for_its_error_rate(self):
        y_true = np.array([1, 1])
        y_pred = np.array([1, 0])

        metrics = compute_metrics(y_true, y_pred)

        self.assertTrue(np.isnan(metrics["apcer"]))
        self.assertAlmostEqual(metrics["bpcer"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
