"""Default paths must point at the ml/ directory, not the module folder.

The defaults used to be derived from each script's own location. Moving
the scripts into capability folders makes that arithmetic wrong unless
it is centralized, so these tests pin the resolved paths.
"""

import unittest
from pathlib import Path

from liveness.paths import (
    DEFAULT_CELEBA_SPOOF_DATA_ROOT,
    DEFAULT_CELEBA_SPOOF_MODEL_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_NUAA_DATA_ROOT,
    ML_ROOT,
)


class PathDefaultsTest(unittest.TestCase):
    def test_ml_root_is_the_ml_directory(self):
        self.assertEqual(ML_ROOT, Path(__file__).resolve().parents[1])

    def test_model_defaults_are_under_ml_root(self):
        self.assertEqual(DEFAULT_MODEL_DIR, ML_ROOT / "models")
        self.assertEqual(DEFAULT_CELEBA_SPOOF_MODEL_DIR, ML_ROOT / "models" / "celeba_spoof")

    def test_dataset_defaults_are_under_ml_data(self):
        self.assertEqual(DEFAULT_NUAA_DATA_ROOT, ML_ROOT / "data" / "raw")
        self.assertEqual(DEFAULT_CELEBA_SPOOF_DATA_ROOT, ML_ROOT / "data" / "celeba_spoof")
