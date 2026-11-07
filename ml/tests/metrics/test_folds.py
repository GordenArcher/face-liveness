"""Fold summaries must report the mean and population std of each metric.

Both cross-validation entry points use this, so a rounding or std
convention change here changes every reported CV number.
"""

import io
import unittest
from contextlib import redirect_stdout

from liveness.metrics.folds import summarize_folds


class SummarizeFoldsTest(unittest.TestCase):
    def test_prints_mean_and_std_for_each_metric(self):
        fold_metrics = [
            {"acer": 0.1, "apcer": 0.2, "bpcer": 0.0},
            {"acer": 0.3, "apcer": 0.4, "bpcer": 0.2},
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            summarize_folds(fold_metrics)

        text = output.getvalue()
        self.assertIn("ACER:  0.2000 +/- 0.1000", text)
        self.assertIn("APCER: 0.3000 +/- 0.1000", text)
        self.assertIn("BPCER: 0.1000 +/- 0.1000", text)
