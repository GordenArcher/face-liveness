"""Loads NUAA's real directory layout: ClientFace/ (live) and
ImposterFace/ (spoof) at the top level, with images grouped into
per-subject subdirectories underneath (e.g. ClientFace/0001/*.jpg).
Recursive glob rather than assuming a flat structure, this matches how
the commonly redistributed "detected face" version of NUAA is actually
organized, not a convention invented for this project.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

LIVE_LABEL = 1
SPOOF_LABEL = 0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sample:
    path: Path
    label: int


def _collect(root: Path, label: int) -> list[Sample]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"expected a directory at {root}, see ml/data/README.md for how to obtain and place the NUAA dataset"
        )
    samples = [
        Sample(path=p, label=label)
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not samples:
        raise ValueError(f"no images found under {root}, is the dataset actually extracted here?")
    return samples


def load_samples(data_root: Path) -> list[Sample]:
    """data_root is expected to contain ClientFace/ and ImposterFace/
    directly, e.g. ml/data/raw/."""
    live = _collect(data_root / "ClientFace", LIVE_LABEL)
    spoof = _collect(data_root / "ImposterFace", SPOOF_LABEL)
    return live + spoof


def train_val_test_split(
    samples: list[Sample], val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
):
    """Stratified on label so live/spoof ratio stays consistent across
    splits, NUAA's two classes aren't evenly sized (more imposter images
    than client images per the dataset's own numbers), an unstratified
    split could easily leave one split short of one class.
    """
    labels = [s.label for s in samples]
    train_val, test = train_test_split(
        samples, test_size=test_size, stratify=labels, random_state=seed
    )
    train_val_labels = [s.label for s in train_val]
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, stratify=train_val_labels, random_state=seed
    )
    return train, val, test


def labels_array(samples: list[Sample]) -> np.ndarray:
    return np.array([s.label for s in samples])