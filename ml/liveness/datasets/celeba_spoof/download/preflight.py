"""Disk-space checks before download, assembly, and extraction.

These run before the expensive steps because an ENOSPC partway through
a multi-hour download is the failure this project actually hit. The
checks are deliberately separate functions: the free-space requirement
for the download directory is different from the requirement for the
volume receiving the assembled archive and extracted files.
"""

import shutil
from pathlib import Path


def assert_enough_free_space(path: Path, min_free_gb: float) -> None:
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024**3)
    if free_gb >= min_free_gb:
        return

    raise SystemExit(
        f"only {free_gb:.1f} GiB free at {path}, but at least {min_free_gb:.1f} GiB is required. "
        "CelebA-Spoof is distributed as many large split zip parts and also needs extraction room. "
        "Free disk space, choose a larger --data-root, or rerun with --skip-space-check if you know this is enough."
    )


def is_no_space_left(path: Path) -> bool:
    try:
        probe = path / ".space-check.tmp"
        with open(probe, "wb") as f:
            f.write(b"0")
        probe.unlink()
        return False
    except OSError as exc:
        return exc.errno == 28


def assert_enough_space_for_combined_archive(combined_path: Path, parts: list[Path]) -> None:
    required_bytes = sum(part.stat().st_size for part in parts)
    free_bytes = shutil.disk_usage(combined_path.parent).free
    if free_bytes >= required_bytes:
        return

    required_gb = required_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)
    raise SystemExit(
        f"all split archive parts are present, but only {free_gb:.1f} GiB is free at {combined_path.parent}. "
        f"assembling {combined_path.name} requires about {required_gb:.1f} GiB before extraction can start. "
        "Move the split parts to a larger volume, free more disk space, or rerun with --no-extract to stop after download."
    )
