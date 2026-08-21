"""Downloads and prepares CelebA-Spoof into this project's data layout.

The official CelebA-Spoof distribution is hosted behind Google Drive
and Baidu Drive links rather than a plain versioned HTTP archive. That
means a downloader is useful, but it should not silently hide the
dataset terms or pretend the external host is as stable as a package
registry. This script does three practical things:

1. Requires the caller to explicitly acknowledge CelebA-Spoof's
   non-commercial research terms before downloading.
2. Uses `gdown` for the official Google Drive folder because standard
   urllib/curl flows often fail on Drive folders and large files.
3. Extracts downloaded zip files and validates that the train/val/test
   annotation JSON files can be found by the loader used for training.
"""

import argparse
from dataclasses import dataclass
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from celeba_spoof_dataset import find_annotation_file

OFFICIAL_GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing"
PROGRESS_INTERVAL_SECONDS = 10.0
USE_COLOR = False


class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


@dataclass(frozen=True)
class SplitArchiveAssembly:
    parts: list[Path]
    combined_archives: list[Path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "celeba_spoof",
        help="where CelebA-Spoof should be downloaded/extracted",
    )
    parser.add_argument(
        "--google-drive-url",
        default=OFFICIAL_GOOGLE_DRIVE_URL,
        help="Google Drive folder URL; defaults to the official CelebA-Spoof folder linked from the project README",
    )
    parser.add_argument(
        "--i-accept-non-commercial-research-terms",
        action="store_true",
        help="required acknowledgement of CelebA-Spoof's non-commercial research dataset agreement",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="skip gdown and only extract/validate files already present under data-root",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="download only; do not assemble, extract, or validate annotation files",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="assemble already downloaded split zip parts, then stop before extraction and validation",
    )
    parser.add_argument(
        "--combined-zip-dir",
        type=Path,
        default=None,
        help="write the assembled CelebA-Spoof zip to this directory instead of beside the split parts",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="keep downloaded zip files and split zip parts after successful extraction",
    )
    parser.add_argument(
        "--strict-folder-limit",
        action="store_true",
        help="do not pass gdown's --remaining-ok flag; useful only if the upstream Drive folder layout changes",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=120.0,
        help="minimum free space required at data-root before download/extraction starts",
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="skip the free-space preflight check",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="control colored terminal logs; auto enables color only for interactive terminals",
    )
    args = parser.parse_args()
    configure_color(args.color)

    require_dataset_terms(args.i_accept_non_commercial_research_terms)
    args.data_root.mkdir(parents=True, exist_ok=True)
    if args.combined_zip_dir is not None:
        args.combined_zip_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_space_check:
        assert_enough_free_space(args.data_root, args.min_free_gb)

    if not args.skip_download:
        download_google_drive_folder(
            args.google_drive_url,
            args.data_root,
            remaining_ok=not args.strict_folder_limit,
        )

    if args.assemble_only:
        assembly = assemble_split_archives(args.data_root, combined_zip_dir=args.combined_zip_dir)
        if not assembly.combined_archives:
            log_status("skip", f"no split zip archive parts found under {args.data_root}", color=Color.YELLOW)
            return
        log_status("done", "assembled combined archive(s):", color=Color.GREEN)
        for archive in assembly.combined_archives:
            print(f"- {archive}")
        log_status("stop", "stopping before extraction because --assemble-only was passed", color=Color.YELLOW)
        return

    if not args.no_extract:
        extract_archives(
            args.data_root,
            keep_zips=args.keep_zips,
            combined_zip_dir=args.combined_zip_dir,
        )

    if args.no_extract:
        log_status(
            "done",
            f"CelebA-Spoof archive parts are downloaded under {args.data_root}. "
            "Rerun without --no-extract on a volume with enough free space to assemble, extract, and validate.",
            color=Color.GREEN,
        )
    else:
        validate_prepared_dataset(args.data_root)
        log_status("ready", f"CelebA-Spoof is ready at {args.data_root}", color=Color.GREEN)


def require_dataset_terms(accepted: bool) -> None:
    if accepted:
        return

    raise SystemExit(
        "CelebA-Spoof is restricted to non-commercial research/education use. "
        "Read the official dataset agreement, then rerun with "
        "--i-accept-non-commercial-research-terms if you accept it."
    )


def download_google_drive_folder(url: str, output_dir: Path, remaining_ok: bool) -> None:
    if importlib.util.find_spec("gdown") is None:
        raise SystemExit(
            "gdown is required for the Google Drive folder download. "
            "Install this repo's requirements first with `python -m pip install -r requirements.txt`."
        )

    # I call gdown through its CLI instead of importing private Python
    # APIs because the CLI is the stable interface users already know
    # from dataset download instructions. `--continue` matters here
    # because the official folder is large enough that interrupted
    # downloads are normal, not exceptional; restarting 50 split archive
    # parts from scratch would waste hours and bandwidth.
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        "--continue",
        url,
        "-O",
        str(output_dir),
    ]
    if remaining_ok:
        # The official CelebA-Spoof Google Drive folder is a split
        # archive (`CelebA_Spoof.zip.001` ... `.050`). gdown treats
        # folders with many files defensively and exits unless this flag
        # is present. Without it, the download stops before any archive
        # can be assembled or validated.
        cmd.append("--remaining-ok")
    log_status("download", "downloading CelebA-Spoof from Google Drive", color=Color.BLUE)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        if is_no_space_left(output_dir):
            raise SystemExit(
                f"download stopped because {output_dir} is out of disk space. "
                "Free space, then rerun the same command; gdown is called with --continue, "
                "so completed and partial split archive files can resume."
            ) from exc
        raise


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


def extract_archives(data_root: Path, keep_zips: bool, combined_zip_dir: Path | None) -> None:
    assembly = assemble_split_archives(data_root, combined_zip_dir=combined_zip_dir)
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
        log_status("extract", f"extracting {archive}", color=Color.BLUE)
        with zipfile.ZipFile(archive) as zf:
            extract_zip_with_progress(zf, data_root)
        if not keep_zips:
            archive.unlink()

    if not keep_zips:
        for part in assembly.parts:
            part.unlink()


def assemble_split_archives(data_root: Path, combined_zip_dir: Path | None) -> SplitArchiveAssembly:
    split_parts = sorted(data_root.rglob("*.zip.[0-9][0-9][0-9]"))
    if not split_parts:
        return SplitArchiveAssembly(parts=[], combined_archives=[])

    groups: dict[Path, list[Path]] = {}
    for part in split_parts:
        groups.setdefault(split_archive_base(part, combined_zip_dir=combined_zip_dir), []).append(part)

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


def print_progress(
    label: str,
    current: int,
    total: int,
    last_progress_at: float,
) -> float:
    now = time.monotonic()
    if now - last_progress_at < PROGRESS_INTERVAL_SECONDS and current < total:
        return last_progress_at

    percent = (current / total * 100) if total else 100.0
    progress_line = (
        f"{paint(label, Color.CYAN)}: "
        f"{paint(f'{percent:6.2f}%', Color.GREEN)} "
        f"{paint(f'({format_bytes(current)} / {format_bytes(total)})', Color.DIM)}"
    )
    print(progress_line, flush=True)
    return now


def configure_color(mode: str) -> None:
    global USE_COLOR

    # Color is useful when watching a long preparation job in the
    # terminal, but raw ANSI escapes make copied logs and redirected files
    # harder to read. `auto` follows the normal CLI convention: color only
    # when stdout is a TTY and the user has not opted out with NO_COLOR.
    if mode == "always":
        USE_COLOR = True
        return
    if mode == "never":
        USE_COLOR = False
        return

    USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{Color.RESET}"


def log_status(label: str, message: str, color: str = Color.CYAN) -> None:
    print(f"{paint(label, color)}: {message}", flush=True)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{size} B"


def split_archive_base(part: Path, combined_zip_dir: Path | None) -> Path:
    combined_name = part.with_suffix("").name
    if combined_zip_dir is None:
        return part.with_suffix("")

    return combined_zip_dir / combined_name


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


def validate_prepared_dataset(data_root: Path) -> None:
    missing = []
    for split in ("train", "val", "test"):
        try:
            annotation_path = find_annotation_file(data_root, split)
            print(f"found {split} annotations: {annotation_path}")
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        raise SystemExit(
            "CelebA-Spoof download/extraction finished, but the expected split annotations were not found:\n"
            + "\n".join(f"- {message}" for message in missing)
        )


if __name__ == "__main__":
    main()
