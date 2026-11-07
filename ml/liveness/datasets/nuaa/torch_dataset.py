"""Torch dataset for a saved NUAA split.

Reads the same `split.json` the baseline trainer wrote, not a freshly
computed split. M1 and M2 need to train and evaluate on the exact same
images for the eventual comparison between them to mean anything, if
each milestone computed its own split independently there'd be no
guarantee they even held out the same test images.
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


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
