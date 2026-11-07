"""Disk-space preflight checks must fail before expensive work starts."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from liveness.datasets.celeba_spoof.download.preflight import (
    assert_enough_free_space,
    assert_enough_space_for_combined_archive,
    is_no_space_left,
)


class FreeSpaceTest(unittest.TestCase):
    def test_zero_requirement_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert_enough_free_space(Path(tmp), 0)

    def test_huge_requirement_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                assert_enough_free_space(Path(tmp), 1e12)

    def test_writable_directory_is_not_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(is_no_space_left(Path(tmp)))

    def test_combined_archive_check_raises_when_volume_is_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = []
            for name in ("part-a", "part-b"):
                part = root / name
                part.write_bytes(b"x" * 10)
                parts.append(part)

            full_usage = type("Usage", (), {"free": 0})()
            with mock.patch(
                "liveness.datasets.celeba_spoof.download.preflight.shutil.disk_usage",
                return_value=full_usage,
            ):
                with self.assertRaises(SystemExit):
                    assert_enough_space_for_combined_archive(root / "combined.zip", parts)
