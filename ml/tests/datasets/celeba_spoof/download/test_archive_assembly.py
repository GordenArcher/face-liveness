"""Split parts must assemble into one valid zip, in order.

The official release is downloaded as `.001` ... `.050` byte-split
parts. A missing or out-of-order part must fail loudly rather than
produce a corrupt combined archive.
"""

import io
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from liveness.datasets.celeba_spoof.download.archive.assemble import assemble_split_archives


def build_zip_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("train/0001.jpg", b"first")
            archive.writestr("train/0002.jpg", b"second")
        return path.read_bytes()


def write_parts(directory: Path, data: bytes, count: int = 2) -> None:
    chunk_size = (len(data) + count - 1) // count
    for index in range(count):
        part = directory / f"CelebA_Spoof.zip.{index + 1:03d}"
        part.write_bytes(data[index * chunk_size:(index + 1) * chunk_size])


class AssembleSplitArchivesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_root = self.root / "data"
        self.output_dir = self.root / "output"
        self.data_root.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _assemble(self):
        # Assembly prints progress and reuse lines; keep the test runner
        # output focused on pass/fail.
        with redirect_stdout(io.StringIO()):
            return assemble_split_archives(self.data_root, self.output_dir)

    def test_assembles_parts_into_a_valid_zip(self):
        write_parts(self.data_root, build_zip_bytes())

        result = self._assemble()
        combined = self.output_dir / "CelebA_Spoof.zip"

        self.assertEqual(result.parts, sorted(self.data_root.glob("*.zip.[0-9][0-9][0-9]")))
        self.assertEqual(result.combined_archives, [combined])
        self.assertTrue(zipfile.is_zipfile(combined))
        with zipfile.ZipFile(combined) as archive:
            self.assertEqual(archive.read("train/0001.jpg"), b"first")
            self.assertEqual(archive.read("train/0002.jpg"), b"second")

    def test_reuses_existing_valid_combined_archive(self):
        write_parts(self.data_root, build_zip_bytes())

        first = self._assemble()
        second = self._assemble()

        self.assertEqual(first.combined_archives, second.combined_archives)

    def test_rebuilds_stale_invalid_combined_archive(self):
        write_parts(self.data_root, build_zip_bytes())
        (self.output_dir / "CelebA_Spoof.zip").write_bytes(b"not a zip")

        result = self._assemble()

        self.assertTrue(zipfile.is_zipfile(result.combined_archives[0]))

    def test_incomplete_parts_raise(self):
        data = build_zip_bytes()
        chunk_size = (len(data) + 1) // 2
        (self.data_root / "CelebA_Spoof.zip.001").write_bytes(data[:chunk_size])
        (self.data_root / "CelebA_Spoof.zip.003").write_bytes(data[chunk_size:])

        with self.assertRaises(SystemExit):
            self._assemble()

    def test_no_parts_returns_empty_result(self):
        result = self._assemble()

        self.assertEqual(result.parts, [])
        self.assertEqual(result.combined_archives, [])
