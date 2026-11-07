"""The CelebA-Spoof torch dataset must match the NUAA dataset contract."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from liveness.datasets.celeba_spoof.samples import CelebASpoofSample
from liveness.datasets.celeba_spoof.torch_dataset import CelebASpoofDataset
from liveness.datasets.transforms import EVAL_TRANSFORM, IMAGE_SIZE


class CelebASpoofDatasetTest(unittest.TestCase):
    def test_item_returns_transformed_image_and_float_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "face.jpg"
            Image.new("RGB", (64, 64), (100, 110, 120)).save(image_path)
            sample = CelebASpoofSample(
                path=image_path,
                label=0,
                subject_id="subject_1",
                spoof_type=1,
            )

            dataset = CelebASpoofDataset([sample], EVAL_TRANSFORM)
            image, label = dataset[0]

            self.assertEqual(tuple(image.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
            self.assertIsInstance(label, float)
            self.assertEqual(label, 0.0)
