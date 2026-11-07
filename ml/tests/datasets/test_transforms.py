"""Shared transforms must produce the tensor shape every model expects."""

import unittest

import torch
from PIL import Image

from liveness.datasets.transforms import EVAL_TRANSFORM, IMAGE_SIZE, TRAIN_TRANSFORM


class TransformsTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 150), (120, 130, 140))

    def test_eval_transform_produces_normalized_tensor(self):
        tensor = EVAL_TRANSFORM(self.image)

        self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())

    def test_train_transform_produces_same_shape(self):
        tensor = TRAIN_TRANSFORM(self.image)

        self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(torch.isfinite(tensor).all())
