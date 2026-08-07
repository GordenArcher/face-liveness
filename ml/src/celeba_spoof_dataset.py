"""CelebA-Spoof loading utilities.

CelebA-Spoof is the next generalization dataset after NUAA. NUAA is
small and mostly print-attack focused; CelebA-Spoof is much larger and
contains photo, replay/screen, and mask-style attacks across many more
subjects. The important implementation detail is that CelebA-Spoof does
not use NUAA's `ClientFace/` and `ImposterFace/` directory convention.
Its official release stores annotations in JSON, with index 40 holding
the spoof type: 0 means live, non-zero values are spoof media.

This module converts that release format into the same binary label
contract the rest of this project already uses: live = 1, spoof = 0.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

LIVE_LABEL = 1
SPOOF_LABEL = 0

SPOOF_TYPE_INDEX = 40


@dataclass(frozen=True)
class CelebASpoofSample:
    path: Path
    label: int
    subject_id: str
    spoof_type: int
    illumination: int | None = None
    environment: int | None = None


class CelebASpoofDataset(Dataset):
    def __init__(self, samples: list[CelebASpoofSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.transform(image)
        return image, float(sample.label)


def load_celeba_spoof_split(data_root: Path, split: str, limit: int | None = None) -> list[CelebASpoofSample]:
    annotation_path = find_annotation_file(data_root, split)

    with open(annotation_path) as f:
        annotations = json.load(f)

    if not isinstance(annotations, dict):
        raise ValueError(
            f"{annotation_path} should be a JSON object mapping image paths to annotation arrays"
        )

    samples = []
    for image_key, values in annotations.items():
        image_path = resolve_image_path(data_root, split, image_key)
        spoof_type = parse_spoof_type(annotation_path, image_key, values)
        label = LIVE_LABEL if spoof_type == 0 else SPOOF_LABEL
        samples.append(
            CelebASpoofSample(
                path=image_path,
                label=label,
                subject_id=infer_subject_id(image_key, image_path),
                spoof_type=spoof_type,
                illumination=parse_optional_int(values, 41),
                environment=parse_optional_int(values, 42),
            )
        )

        if limit is not None and len(samples) >= limit:
            break

    assert_both_labels(split, samples)
    return samples


def find_annotation_file(data_root: Path, split: str) -> Path:
    split_aliases = {
        "val": ["val", "valid", "validation"],
        "valid": ["valid", "val", "validation"],
        "validation": ["validation", "val", "valid"],
    }
    names = split_aliases.get(split, [split])
    candidates = [
        candidate
        for name in names
        for candidate in (
            data_root / f"{name}_label.json",
            data_root / f"{name}_labels.json",
            data_root / f"{name}.json",
            data_root / "metas" / f"{name}_label.json",
            data_root / "metas" / f"{name}_labels.json",
            data_root / "annotations" / f"{name}_label.json",
            data_root / "annotations" / f"{name}_labels.json",
        )
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(
        match
        for name in names
        for match in data_root.rglob(f"{name}*label*.json")
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"could not find a CelebA-Spoof annotation JSON for split {split!r} under {data_root}. "
        f"Expected names like {split}_label.json or {split}_labels.json."
    )


def resolve_image_path(data_root: Path, split: str, image_key: str) -> Path:
    raw = Path(image_key)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                data_root / raw,
                data_root / split / raw,
                data_root / "Data" / split / raw,
                data_root / "data" / split / raw,
                data_root / "images" / split / raw,
                data_root / split / "images" / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"annotation references {image_key!r}, but none of the expected image locations exist under {data_root}"
    )


def parse_spoof_type(annotation_path: Path, image_key: str, values) -> int:
    if not isinstance(values, list) or len(values) <= SPOOF_TYPE_INDEX:
        raise ValueError(
            f"{annotation_path}:{image_key} should contain at least {SPOOF_TYPE_INDEX + 1} annotation values"
        )
    return int(values[SPOOF_TYPE_INDEX])


def parse_optional_int(values, index: int) -> int | None:
    if isinstance(values, list) and len(values) > index:
        return int(values[index])
    return None


def infer_subject_id(image_key: str, image_path: Path) -> str:
    parts = Path(image_key).parts
    if len(parts) >= 2:
        return parts[-2]
    return image_path.parent.name


def assert_both_labels(name: str, samples: list[CelebASpoofSample]) -> None:
    labels = {sample.label for sample in samples}
    if labels != {LIVE_LABEL, SPOOF_LABEL}:
        raise ValueError(
            f"{name} split contains label(s) {labels}, not both live and spoof. "
            "Check that the JSON annotations are the official CelebA-Spoof format."
        )


def summarize_samples(name: str, samples: list[CelebASpoofSample]) -> str:
    live = sum(1 for sample in samples if sample.label == LIVE_LABEL)
    spoof = sum(1 for sample in samples if sample.label == SPOOF_LABEL)
    subjects = len({sample.subject_id for sample in samples})
    return f"{name}: {len(samples)} images, {subjects} subjects, {live} live, {spoof} spoof"
