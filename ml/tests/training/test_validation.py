"""Shared epoch validation must select checkpoints by the same criteria.

Both the NUAA and CelebA-Spoof trainers call this. The test uses a fixed
logit so the expected APCER/BPCER/ACER are known exactly.
"""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from liveness.training.validation import run_validation


class FixedLogitModel(nn.Module):
    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.size(0),), self.logit)


class RunValidationTest(unittest.TestCase):
    def test_returns_loss_and_metrics(self):
        images = torch.zeros(4, 3, 8, 8)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        # sigmoid(0.0) == 0.5, so every sample lands exactly on the live
        # side of the 0.5 cutoff. Both spoofs are therefore accepted.
        loss, metrics = run_validation(FixedLogitModel(0.0), loader, torch.device("cpu"))

        self.assertAlmostEqual(loss, 0.6931, places=4)
        self.assertAlmostEqual(metrics["apcer"], 1.0)
        self.assertAlmostEqual(metrics["bpcer"], 0.0)
        self.assertAlmostEqual(metrics["acer"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
