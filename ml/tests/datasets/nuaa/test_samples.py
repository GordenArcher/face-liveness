"""NUAA sample loading must keep the folder-to-label mapping intact.

The subject ID is taken from the immediate parent directory; that is
what makes subject-disjoint splitting possible later.
"""

import tempfile
import unittest
from pathlib import Path

from liveness.datasets.nuaa.samples import (
    LIVE_LABEL,
    SPOOF_LABEL,
    Sample,
    labels_array,
    load_samples,
)


class LoadSamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ClientFace" / "0001").mkdir(parents=True)
        (self.root / "ClientFace" / "0002").mkdir(parents=True)
        (self.root / "ImposterFace" / "0001").mkdir(parents=True)
        (self.root / "ClientFace" / "0001" / "live.jpg").touch()
        (self.root / "ClientFace" / "0002" / "live.PNG").touch()
        (self.root / "ImposterFace" / "0001" / "spoof.jpeg").touch()
        (self.root / "ImposterFace" / "0001" / "notes.txt").touch()

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_both_folders_with_subject_ids(self):
        samples = load_samples(self.root)
        by_name = {sample.path.name: sample for sample in samples}

        self.assertEqual(len(samples), 3)
        self.assertEqual(by_name["live.jpg"].label, LIVE_LABEL)
        self.assertEqual(by_name["live.jpg"].subject_id, "0001")
        self.assertEqual(by_name["live.PNG"].label, LIVE_LABEL)
        self.assertEqual(by_name["spoof.jpeg"].label, SPOOF_LABEL)
        self.assertEqual(by_name["spoof.jpeg"].subject_id, "0001")

    def test_missing_dataset_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_samples(self.root / "missing")

    def test_empty_dataset_folder_raises(self):
        empty = self.root / "empty"
        (empty / "ClientFace").mkdir(parents=True)
        (empty / "ImposterFace").mkdir(parents=True)

        with self.assertRaises(ValueError):
            load_samples(empty)

    def test_labels_array_returns_integer_labels(self):
        samples = [
            Sample(Path("live.jpg"), LIVE_LABEL, "0001"),
            Sample(Path("spoof.jpg"), SPOOF_LABEL, "0002"),
        ]

        self.assertEqual(labels_array(samples).tolist(), [1, 0])
