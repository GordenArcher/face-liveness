"""Assembly of CelebA-Spoof's split zip parts.

gdown downloads the official release as `CelebA_Spoof.zip.001` ...
`.050` byte-split parts. They have to be streamed into one normal zip
before Python's `zipfile` module can read them.
"""

import zipfile
from pathlib import Path

from liveness.datasets.celeba_spoof.download.archive.model import SplitArchiveAssembly
from liveness.datasets.celeba_spoof.download.preflight import assert_enough_space_for_combined_archive
from liveness.datasets.celeba_spoof.download.progress import Color, log_status, print_progress


def assemble_split_archives(data_root: Path, output_dir: Path) -> SplitArchiveAssembly:
    split_parts = sorted(data_root.rglob("*.zip.[0-9][0-9][0-9]"))
    if not split_parts:
        return SplitArchiveAssembly(parts=[], combined_archives=[])

    groups: dict[Path, list[Path]] = {}
    for part in split_parts:
        groups.setdefault(output_dir / part.with_suffix("").name, []).append(part)

    combined_archives: list[Path] = []

    for combined_path, parts in groups.items():
        expected_suffixes = [f"{i:03d}" for i in range(1, len(parts) + 1)]
        actual_suffixes = [part.suffix.lstrip(".") for part in parts]
        if actual_suffixes != expected_suffixes:
            raise SystemExit(
                f"split archive parts for {combined_path.name} are incomplete or out of order: "
                f"expected suffixes {expected_suffixes[0]}..{expected_suffixes[-1]}, got {actual_suffixes}"
            )

        if combined_path.exists():
            if zipfile.is_zipfile(combined_path):
                log_status("reuse", f"combined archive already exists: {combined_path}", color=Color.GREEN)
                combined_archives.append(combined_path)
                continue

            log_status("remove", f"removing stale invalid combined archive: {combined_path}", color=Color.YELLOW)
            combined_path.unlink()

        assert_enough_space_for_combined_archive(combined_path, parts)

        temp_path = combined_path.with_name(f"{combined_path.name}.part")
        if temp_path.exists():
            log_status("remove", f"removing stale partial combined archive: {temp_path}", color=Color.YELLOW)
            temp_path.unlink()

        log_status(
            "archive",
            f"assembling {combined_path} from {len(parts)} split archive parts",
            color=Color.BLUE,
        )
        # The upstream files are byte-split zip parts, so I stream them
        # into one normal .zip before using Python's zipfile module. This
        # avoids shell-specific `cat` assumptions and keeps the recovery
        # path readable if one numbered part is missing. The destination
        # can be a different volume from `data_root`; this is important on
        # small internal disks because the 50 downloaded parts may already
        # fill most of the machine, while an external drive can still hold
        # the temporary combined archive. I write to a `.part` file first
        # and rename only after every split part is copied, otherwise an
        # interrupted run leaves a corrupt final `CelebA_Spoof.zip` that
        # looks like the real archive name and causes confusing extraction
        # failures on the next step.
        with open(temp_path, "wb") as combined:
            copied_bytes = 0
            total_bytes = sum(part.stat().st_size for part in parts)
            last_progress_at = 0.0
            for part in parts:
                with open(part, "rb") as source:
                    while chunk := source.read(1024 * 1024):
                        combined.write(chunk)
                        copied_bytes += len(chunk)
                        last_progress_at = print_progress(
                            label="assembly",
                            current=copied_bytes,
                            total=total_bytes,
                            last_progress_at=last_progress_at,
                        )
        temp_path.replace(combined_path)

        combined_archives.append(combined_path)

    return SplitArchiveAssembly(parts=split_parts, combined_archives=combined_archives)
