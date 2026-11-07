"""Subject-disjoint splits must never leak a subject across splits.

This is the regression test for the original bug: an image-level split
let the same person appear in train and test, and the model learned the
person instead of the live-vs-spoof cue.
"""

import unittest
from pathlib import Path

from liveness.datasets.nuaa.samples import LIVE_LABEL, SPOOF_LABEL, Sample
from liveness.datasets.nuaa.split import (
    _assert_both_labels_present,
    train_val_split,
    train_val_test_split,
)


def make_samples(n_subjects: int = 20) -> list[Sample]:
    return [
        Sample(Path(f"{subject}-{label}.jpg"), label, f"{subject:04d}")
        for subject in range(n_subjects)
        for label in (LIVE_LABEL, SPOOF_LABEL)
    ]


class TrainValTestSplitTest(unittest.TestCase):
    def setUp(self):
        self.samples = make_samples()

    def test_splits_are_subject_disjoint_and_cover_all_subjects(self):
        train, val, test = train_val_test_split(self.samples, seed=42)
        train_subjects = {sample.subject_id for sample in train}
        val_subjects = {sample.subject_id for sample in val}
        test_subjects = {sample.subject_id for sample in test}
        all_subjects = {sample.subject_id for sample in self.samples}

        self.assertFalse(train_subjects & val_subjects)
        self.assertFalse(train_subjects & test_subjects)
        self.assertFalse(val_subjects & test_subjects)
        self.assertEqual(train_subjects | val_subjects | test_subjects, all_subjects)

    def test_each_split_has_both_labels(self):
        for split in train_val_test_split(self.samples, seed=42):
            self.assertEqual({sample.label for sample in split}, {LIVE_LABEL, SPOOF_LABEL})

    def test_same_seed_is_deterministic(self):
        first = train_val_test_split(self.samples, seed=42)
        second = train_val_test_split(self.samples, seed=42)

        for first_split, second_split in zip(first, second):
            self.assertEqual(
                {sample.subject_id for sample in first_split},
                {sample.subject_id for sample in second_split},
            )

    def test_train_val_split_is_subject_disjoint(self):
        train, val = train_val_split(self.samples, seed=42)

        self.assertFalse(
            {sample.subject_id for sample in train} & {sample.subject_id for sample in val}
        )
        self.assertEqual(len(train) + len(val), len(self.samples))

    def test_single_label_split_is_rejected(self):
        with self.assertRaises(ValueError):
            _assert_both_labels_present(
                "train",
                [Sample(Path("live.jpg"), LIVE_LABEL, "0001")],
            )
