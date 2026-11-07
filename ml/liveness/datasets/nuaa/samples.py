"""Loads NUAA's real directory layout: ClientFace/ (live) and
ImposterFace/ (spoof) at the top level, with images grouped into
per-subject subdirectories underneath (e.g. ClientFace/0001/*.jpg).
Recursive glob rather than assuming a flat structure, this matches how
the commonly redistributed "detected face" version of NUAA is actually
organized, not a convention invented for this project.

The split logic lives in `split.py` because subject-disjoint splitting
is a separate concern from reading the dataset, and both the regular
split and k-fold need the same `Sample` shape.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIVE_LABEL = 1
SPOOF_LABEL = 0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sample:
    path: Path
    label: int
    subject_id: str


def _collect(root: Path, label: int) -> list[Sample]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"expected a directory at {root}, see ml/data/README.md for how to obtain and place the NUAA dataset"
        )
    samples = [
        # Immediate parent directory name is the subject ID in NUAA's
        # layout (ClientFace/0007/foo.jpg -> subject "0007"), this is
        # what makes subject-disjoint splitting possible without a
        # separate identity lookup table.
        Sample(path=p, label=label, subject_id=p.parent.name)
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


def labels_array(samples: list[Sample]) -> np.ndarray:
    return np.array([s.label for s in samples])
