"""Extraction of downloaded CelebA-Spoof archives.

The official release is published as `CelebA_Spoof.zip.001` ...
`.050` byte-split parts. Once `assemble.py` has streamed them into one
normal zip, this module extracts it one member at a time so long runs
keep reporting progress.
"""

import zipfile
from pathlib import Path

from liveness.datasets.celeba_spoof.download.archive.assemble import assemble_split_archives
from liveness.datasets.celeba_spoof.download.progress import Color, log_status, print_progress


def extract_archives(data_root: Path, keep_zips: bool, output_dir: Path) -> None:
    assembly = assemble_split_archives(data_root, output_dir=output_dir)
    archives = sorted(data_root.rglob("*.zip"))
    archive_paths = set(archives)
    for archive in assembly.combined_archives:
        if archive not in archive_paths:
            archives.append(archive)
            archive_paths.add(archive)

    if not archives:
        log_status("skip", f"no zip archives found under {data_root}, skipping extraction", color=Color.YELLOW)
        return

    for archive in archives:
        log_status("extract", f"extracting {archive} into {output_dir}", color=Color.BLUE)
        try:
            with zipfile.ZipFile(archive) as zf:
                extract_zip_with_progress(zf, output_dir)
        except OSError as exc:
            if exc.errno == 28:
                # Caught here instead of via the write-probe trick used for the
                # gdown subprocess, because this exception already carries the
                # real errno. archive is intentionally left in place (we only
                # unlink after a successful extraction below), so rerunning the
                # same command after freeing space, or pointing --output-dir at
                # a larger volume, picks back up without re-downloading.
                raise SystemExit(
                    f"extraction stopped because {output_dir} ran out of disk space "
                    f"while extracting {archive.name}. {archive.name} was kept so you can "
                    "retry after freeing space, or rerun with a larger --output-dir; files "
                    f"already extracted under {output_dir} do not need to be removed first."
                ) from exc
            raise
        if not keep_zips:
            archive.unlink()

    if not keep_zips:
        for part in assembly.parts:
            part.unlink()


def extract_zip_with_progress(zf: zipfile.ZipFile, output_dir: Path) -> None:
    members = zf.infolist()
    total_bytes = sum(member.file_size for member in members)
    extracted_bytes = 0
    last_progress_at = 0.0

    # `ZipFile.extractall()` is convenient, but it is a poor fit for this
    # dataset because CelebA-Spoof can spend hours writing a very large
    # archive without printing anything. I extract one member at a time so
    # long runs keep reporting both file-count and byte-count progress.
    # That makes it clear whether the process is still moving, blocked on a
    # slow external disk, or about to fail because the target volume stopped
    # accepting writes.
    for index, member in enumerate(members, start=1):
        zf.extract(member, output_dir)
        extracted_bytes += member.file_size
        last_progress_at = print_progress(
            label=f"extraction {index}/{len(members)} files",
            current=extracted_bytes,
            total=total_bytes,
            last_progress_at=last_progress_at,
        )
