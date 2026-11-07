"""Torch dataset for CelebA-Spoof samples.

Kept separate from `samples.py` because annotation parsing and torch
batching are different concerns. Both datasets expose the same
`(image, float_label)` contract so the training and evaluation loops do
not need dataset-specific branches.
"""

from PIL import Image
from torch.utils.data import Dataset

from liveness.datasets.celeba_spoof.samples import CelebASpoofSample


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
