# ml/src/cnn_dataset.py
"""Reads the same split.json train_baseline.py wrote, not a freshly
computed split. M1 and M2 need to train and evaluate on the exact same
images for the eventual comparison between them to mean anything, if
each milestone computed its own split independently there'd be no
guarantee they even held out the same test images.
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 128

# Augmentation only, normalization stats below are ImageNet's mean/std.
# Not because this model is pretrained on ImageNet, it isn't, but because
# they're a reasonable, well-tested default for natural face images
# rather than an arbitrary choice, revisit if training instability shows
# up and this turns out to matter.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Small, realistic augmentations for a webcam-captured dataset,
        # not aggressive ones, NUAA's live/spoof distinction is a subtle
        # texture cue, augmentation strong enough to distort that cue
        # would hurt more than it helps.
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)


class SplitDataset(Dataset):
    def __init__(self, entries: list[dict], transform):
        self.entries = entries
        self.transform = transform

    @classmethod
    def from_split_file(cls, split_path: Path, split_name: str, transform) -> "SplitDataset":
        with open(split_path) as f:
            record = json.load(f)
        return cls(record[split_name], transform)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry["path"]).convert("RGB")
        image = self.transform(image)
        # float32 to match BCEWithLogitsLoss's expected target dtype,
        # an int label here is a common, easy-to-miss source of a
        # confusing dtype-mismatch error at loss.backward() time.
        label = float(entry["label"])
        return image, label