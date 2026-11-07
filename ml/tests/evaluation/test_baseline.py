"""The evaluator must read the split training saved, not recompute one."""

import json
import tempfile
import unittest
from pathlib import Path

from liveness.evaluation.baseline import load_test_split


class LoadTestSplitTest(unittest.TestCase):
    def test_reads_saved_test_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "split.json").write_text(
                json.dumps(
                    {
                        "test": [
                            {"path": "/data/ClientFace/0001/face.jpg", "label": 1},
                        ]
                    }
                )
            )

            samples = load_test_split(model_dir)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].label, 1)
            self.assertEqual(samples[0].subject_id, "0001")

    def test_missing_split_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_test_split(Path(tmp))
