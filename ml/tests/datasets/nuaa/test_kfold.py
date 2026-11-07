"""GroupKFold folds must use subjects as groups, not individual images.

Every subject should appear in exactly one held-out fold across the
runs, and each fold must still contain both labels.
"""

import unittest
from pathlib import Path

from liveness.datasets.nuaa.kfold import make_folds
from liveness.datasets.nuaa.samples import LIVE_LABEL, SPOOF_LABEL, Sample


def make_samples(n_subjects: int = 20) -> list[Sample]:
    return [
        Sample(Path(f"{subject}-{label}.jpg"), label, f"{subject:04d}")
        for subject in range(n_subjects)
        for label in (LIVE_LABEL, SPOOF_LABEL)
    ]


class MakeFoldsTest(unittest.TestCase):
    def test_every_subject_is_held_out_exactly_once(self):
        samples = make_samples()
        folds = make_folds(samples, 5)
        all_subjects = {sample.subject_id for sample in samples}
        held_out_subjects = []

        for train, test in folds:
            train_subjects = {sample.subject_id for sample in train}
            test_subjects = {sample.subject_id for sample in test}

            self.assertFalse(train_subjects & test_subjects)
            held_out_subjects.extend(test_subjects)

        self.assertEqual(len(folds), 5)
        self.assertEqual(set(held_out_subjects), all_subjects)
        self.assertEqual(len(held_out_subjects), len(all_subjects))

    def test_each_fold_has_both_labels(self):
        for train, test in make_folds(make_samples(), 5):
            self.assertEqual({sample.label for sample in train}, {LIVE_LABEL, SPOOF_LABEL})
            self.assertEqual({sample.label for sample in test}, {LIVE_LABEL, SPOOF_LABEL})
