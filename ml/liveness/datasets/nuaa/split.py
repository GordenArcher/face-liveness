"""Subject-disjoint splits for NUAA.

Every split in this module is subject-disjoint, not image-disjoint.
NUAA numbers subjects identically across both folders, ClientFace/0007
and ImposterFace/0007 are the same person's live photos and photos of a
spoof attack against them. Splitting at the image level lets the same
subject's photos land in both train and test, a model can then partly
learn "do I recognize this person" instead of the actual live-vs-spoof
texture cue, a shortcut that doesn't exist at real verification time
against someone the model has never seen. An earlier version of this
project split at the image level and produced a suspicious 0.0000
validation ACER, that number was the split leaking, not the model being
good.
"""

import random

from liveness.datasets.nuaa.samples import LIVE_LABEL, SPOOF_LABEL, Sample


def _split_subjects(subject_ids: list[str], fractions: list[float], seed: int) -> list[set]:
    """Divides subject_ids into len(fractions) groups, sized
    proportionally. One implementation shared by every split function
    here (the plain train/val/test split, k-fold's inner train/val
    split), rather than each reimplementing "divide subjects into N
    groups" separately and risking the logic drifting apart between
    them.
    """
    rng = random.Random(seed)
    ids = sorted(subject_ids)  # sorted first so the shuffle is deterministic for a given input
    rng.shuffle(ids)

    n = len(ids)
    sizes = [max(1, round(n * f)) for f in fractions]
    # Last group absorbs whatever rounding left over, so the groups
    # always sum to exactly n instead of silently dropping a subject.
    sizes[-1] = n - sum(sizes[:-1])
    if sizes[-1] < 1:
        raise ValueError(
            f"only {n} subjects total, fractions {fractions} don't leave enough for every group, "
            "use a smaller number of groups or different fractions"
        )

    groups = []
    idx = 0
    for size in sizes:
        groups.append(set(ids[idx:idx + size]))
        idx += size
    return groups


def _assert_both_labels_present(name: str, split: list[Sample]) -> None:
    labels = {s.label for s in split}
    if labels != {LIVE_LABEL, SPOOF_LABEL}:
        raise ValueError(
            f"{name} split only contains label(s) {labels}, not both live and spoof, "
            "the random subject assignment got unlucky, try a different seed"
        )


def train_val_test_split(
    samples: list[Sample], val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
):
    """Splits by subject, no subject's images appear in more than one of
    train/val/test. NUAA only has a handful of subjects total (low
    double digits), so this produces a small number of test subjects,
    that's a real, honest limitation of this dataset, not something to
    paper over by going back to an image-level split that would look
    better and mean less.
    """
    subjects = sorted({s.subject_id for s in samples})
    train_frac = 1 - val_size - test_size
    train_subj, val_subj, test_subj = _split_subjects(subjects, [train_frac, val_size, test_size], seed)

    train = [s for s in samples if s.subject_id in train_subj]
    val = [s for s in samples if s.subject_id in val_subj]
    test = [s for s in samples if s.subject_id in test_subj]

    for name, split in (("train", train), ("val", val), ("test", test)):
        _assert_both_labels_present(name, split)

    return train, val, test


def train_val_split(samples: list[Sample], val_size: float = 0.15, seed: int = 42):
    """Two-way subject split, used by k-fold cross-validation to carve a
    validation set out of a fold's training subjects for epoch
    selection, without touching that fold's held-out test subjects.
    """
    subjects = sorted({s.subject_id for s in samples})
    train_subj, val_subj = _split_subjects(subjects, [1 - val_size, val_size], seed)

    train = [s for s in samples if s.subject_id in train_subj]
    val = [s for s in samples if s.subject_id in val_subj]

    for name, split in (("train", train), ("val", val)):
        _assert_both_labels_present(name, split)

    return train, val
