"""The saved-split dataset must return the training loop's contract.

The CNN loss expects float targets, and the image tensor must match the
shared transform size.
"""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from liveness.datasets.nuaa.torch_dataset import SplitDataset
from liveness.datasets.transforms import EVAL_TRANSFORM, IMAGE_SIZE


class SplitDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image_path = self.root / "face.jpg"
        Image.new("RGB", (64, 64), (100, 110, 120)).save(self.image_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_item_returns_transformed_image_and_float_label(self):
        dataset = SplitDataset([{"path": str(self.image_path), "label": 1}], EVAL_TRANSFORM)

        image, label = dataset[0]

        self.assertEqual(tuple(image.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertIsInstance(label, float)
        self.assertEqual(label, 1.0)

    def test_from_split_file_reads_named_split(self):
        split_path = self.root / "split.json"
        split_path.write_text(
            json.dumps(
                {
                    "train": [{"path": str(self.image_path), "label": 0}],
                    "val": [],
                }
            )
        )

        dataset = SplitDataset.from_split_file(split_path, "train", EVAL_TRANSFORM)

        self.assertEqual(len(dataset), 1)
        _, label = dataset[0]
        self.assertEqual(label, 0.0)
