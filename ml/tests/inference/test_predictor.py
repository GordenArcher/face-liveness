"""The liveness gate's result contract and loader errors.

The future service depends on this shape, so the pass/fail boundary is
pinned here independently of any trained model artifact.
"""

import unittest
from pathlib import Path

from liveness.inference.predictor import _result, load_predictor


class LivenessResultTest(unittest.TestCase):
    def test_score_at_threshold_passes(self):
        result = _result("cnn", live_score=0.5, threshold=0.5)

        self.assertTrue(result.passed)
        self.assertEqual(result.label, "live")
        self.assertAlmostEqual(result.spoof_score, 0.5)

    def test_score_below_threshold_rejects(self):
        result = _result("cnn", live_score=0.4, threshold=0.5)

        self.assertFalse(result.passed)
        self.assertEqual(result.label, "spoof")
        self.assertAlmostEqual(result.spoof_score, 0.6)


class LoadPredictorTest(unittest.TestCase):
    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            load_predictor("unknown", Path("."))

    def test_missing_cnn_weights_raise_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_predictor("cnn", Path("does-not-exist"))

    def test_missing_baseline_artifacts_raise_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_predictor("baseline", Path("does-not-exist"))
