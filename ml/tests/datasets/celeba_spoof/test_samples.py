"""CelebA-Spoof annotation parsing must map spoof type 0 to live.

The official release stores annotations as JSON arrays with spoof type
at index 40. Getting that index or the label polarity wrong would invert
the dataset.
"""

import json
import tempfile
import unittest
from pathlib import Path

from liveness.datasets.celeba_spoof.samples import (
    LIVE_LABEL,
    SPOOF_LABEL,
    CelebASpoofSample,
    assert_both_labels,
    find_annotation_file,
    load_celeba_spoof_split,
    parse_spoof_type,
    resolve_image_path,
    summarize_samples,
)


def annotation(spoof_type: int, illumination: int = 2, environment: int = 3) -> list[int]:
    values = [0] * 43
    values[40] = spoof_type
    values[41] = illumination
    values[42] = environment
    return values


class CelebASpoofSamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        image_dir = self.root / "train" / "subject_1"
        image_dir.mkdir(parents=True)
        for name in ("live_1.jpg", "spoof_1.jpg", "live_2.jpg", "spoof_2.jpg"):
            (image_dir / name).touch()
        (self.root / "train_label.json").write_text(
            json.dumps(
                {
                    "subject_1/live_1.jpg": annotation(0),
                    "subject_1/spoof_1.jpg": annotation(1),
                    "subject_1/live_2.jpg": annotation(0),
                    "subject_1/spoof_2.jpg": annotation(1),
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_official_annotation_format(self):
        samples = load_celeba_spoof_split(self.root, "train")
        by_name = {sample.path.name: sample for sample in samples}

        self.assertEqual(len(samples), 4)
        self.assertEqual(by_name["live_1.jpg"].label, LIVE_LABEL)
        self.assertEqual(by_name["spoof_1.jpg"].label, SPOOF_LABEL)
        self.assertEqual(by_name["live_1.jpg"].subject_id, "subject_1")
        self.assertEqual(by_name["live_1.jpg"].spoof_type, 0)
        self.assertEqual(by_name["spoof_1.jpg"].illumination, 2)
        self.assertEqual(by_name["spoof_1.jpg"].environment, 3)

    def test_limit_stops_after_n_samples(self):
        samples = load_celeba_spoof_split(self.root, "train", limit=2)

        self.assertEqual(len(samples), 2)
        self.assertEqual({sample.label for sample in samples}, {LIVE_LABEL, SPOOF_LABEL})

    def test_missing_annotation_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_celeba_spoof_split(self.root, "test")

    def test_val_alias_finds_validation_file(self):
        validation_path = self.root / "val_label.json"
        validation_path.write_text("{}")

        self.assertEqual(find_annotation_file(self.root, "validation"), validation_path)

    def test_resolve_image_path_rejects_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            resolve_image_path(self.root, "train", "subject_1/missing.jpg")

    def test_parse_spoof_type_rejects_short_annotations(self):
        with self.assertRaises(ValueError):
            parse_spoof_type(self.root / "train_label.json", "subject_1/live.jpg", [0] * 10)

    def test_assert_both_labels_rejects_single_label(self):
        samples = [
            CelebASpoofSample(path=Path("live.jpg"), label=LIVE_LABEL, subject_id="s", spoof_type=0)
        ]

        with self.assertRaises(ValueError):
            assert_both_labels("train", samples)

    def test_summarize_samples_counts_labels_and_subjects(self):
        samples = load_celeba_spoof_split(self.root, "train")
        summary = summarize_samples("train", samples)

        self.assertIn("4 images", summary)
        self.assertIn("1 subjects", summary)
        self.assertIn("2 live", summary)
        self.assertIn("2 spoof", summary)
