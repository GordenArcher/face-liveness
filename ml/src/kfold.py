"""Subject-disjoint k-fold splits via sklearn's GroupKFold, group here
means subject_id, not image, for the same reason dataset.py's regular
split is subject-based: an image-level split lets a model partly learn
"which specific person is this" instead of the actual live-vs-spoof
texture cue.

Every subject serves as test data in exactly one fold across the k runs,
so cross-validation uses the whole dataset for evaluation instead of
permanently reserving a slice of NUAA's already-small subject pool as a
holdout that never contributes to any fold's training set.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

from dataset import LIVE_LABEL, SPOOF_LABEL, Sample


def make_folds(samples: list[Sample], n_splits: int) -> list[tuple[list[Sample], list[Sample]]]:
    groups = np.array([s.subject_id for s in samples])
    labels = np.array([s.label for s in samples])

    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in gkf.split(samples, labels, groups):
        train = [samples[i] for i in train_idx]
        test = [samples[i] for i in test_idx]

        for name, split in (("train", train), ("test", test)):
            split_labels = {s.label for s in split}
            if split_labels != {LIVE_LABEL, SPOOF_LABEL}:
                raise ValueError(
                    f"a fold's {name} split only contains label(s) {split_labels}, "
                    f"not both live and spoof, this NUAA subject pool is too small for "
                    f"n_splits={n_splits}, try fewer folds"
                )
        folds.append((train, test))
    return folds


def summarize_folds(fold_metrics: list[dict]) -> None:
    """Shared by cross_validate_baseline.py and cross_validate_cnn.py, one
    implementation of "print mean +/- std across folds" rather than two
    copies that could report the summary statistic differently without
    anyone noticing.
    """
    acers = np.array([m["acer"] for m in fold_metrics])
    apcers = np.array([m["apcer"] for m in fold_metrics])
    bpcers = np.array([m["bpcer"] for m in fold_metrics])

    print("\n=== cross-validation summary ===")
    print(f"ACER:  {acers.mean():.4f} +/- {acers.std():.4f}")
    print(f"APCER: {apcers.mean():.4f} +/- {apcers.std():.4f}")
    print(f"BPCER: {bpcers.mean():.4f} +/- {bpcers.std():.4f}")