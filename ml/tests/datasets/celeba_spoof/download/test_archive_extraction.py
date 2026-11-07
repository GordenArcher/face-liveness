"""Extraction must write members and respect the keep-zips flag."""

import io
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from liveness.datasets.celeba_spoof.download.archive.extract import (
    extract_archives,
    extract_zip_with_progress,
)


class ExtractZipWithProgressTest(unittest.TestCase):
    def test_writes_every_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "sample.zip"
            output_dir = root / "out"
            output_dir.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("train/0001.jpg", b"first")

            with zipfile.ZipFile(archive_path) as archive:
                with redirect_stdout(io.StringIO()):
                    extract_zip_with_progress(archive, output_dir)

            self.assertEqual((output_dir / "train/0001.jpg").read_bytes(), b"first")


class ExtractArchivesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_root = self.root / "data"
        self.output_dir = self.root / "output"
        self.data_root.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_archive(self) -> Path:
        archive_path = self.data_root / "sample.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("train/a.txt", "hello")
        return archive_path

    def _extract(self, keep_zips: bool) -> None:
        # Extraction prints per-file progress; keep the test runner output
        # focused on pass/fail.
        with redirect_stdout(io.StringIO()):
            extract_archives(self.data_root, keep_zips=keep_zips, output_dir=self.output_dir)

    def test_extracts_normal_zip_and_removes_it(self):
        archive_path = self._write_archive()

        self._extract(keep_zips=False)

        self.assertEqual((self.output_dir / "train" / "a.txt").read_text(), "hello")
        self.assertFalse(archive_path.exists())

    def test_keep_zips_preserves_archive(self):
        archive_path = self._write_archive()

        self._extract(keep_zips=True)

        self.assertEqual((self.output_dir / "train" / "a.txt").read_text(), "hello")
        self.assertTrue(archive_path.exists())

    def test_no_archives_is_a_noop(self):
        self._extract(keep_zips=False)

        self.assertEqual(list(self.output_dir.iterdir()), [])
