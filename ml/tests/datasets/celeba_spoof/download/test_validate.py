"""Post-extraction validation must confirm every split annotation exists."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from liveness.datasets.celeba_spoof.download.validate import validate_prepared_dataset


class ValidatePreparedDatasetTest(unittest.TestCase):
    def test_accepts_directory_with_all_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                (root / f"{split}_label.json").write_text(json.dumps({}))

            output = io.StringIO()
            with redirect_stdout(output):
                validate_prepared_dataset(root)

            self.assertIn("found train annotations", output.getvalue())
            self.assertIn("found val annotations", output.getvalue())
            self.assertIn("found test annotations", output.getvalue())

    def test_raises_when_a_split_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train_label.json").write_text("{}")

            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    validate_prepared_dataset(root)
