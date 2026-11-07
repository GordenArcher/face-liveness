"""Torch device selection shared by training, evaluation, and inference.

This used to live in the CNN trainer, which meant evaluation and
inference imported a training script just to pick a device. Keeping it
in its own module removes that accidental dependency direction.
"""

import torch


def select_device() -> torch.device:
    # mps first, this is routinely run on Apple Silicon, falling back to
    # cpu on an M-series Mac would be a silent multi-x slowdown nobody
    # would notice the reason for.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
