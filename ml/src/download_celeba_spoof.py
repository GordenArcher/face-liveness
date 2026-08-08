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
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from celeba_spoof_dataset import find_annotation_file

OFFICIAL_GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing"


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
        help="download/validate only; do not extract zip files",
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
    args = parser.parse_args()

    require_dataset_terms(args.i_accept_non_commercial_research_terms)
    args.data_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_space_check:
        assert_enough_free_space(args.data_root, args.min_free_gb)

    if not args.skip_download:
        download_google_drive_folder(
            args.google_drive_url,
            args.data_root,
            remaining_ok=not args.strict_folder_limit,
        )

    if not args.no_extract:
        extract_archives(args.data_root, keep_zips=args.keep_zips)

    validate_prepared_dataset(args.data_root)
    print(f"CelebA-Spoof is ready at {args.data_root}")


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
    print("downloading CelebA-Spoof from Google Drive")
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


def extract_archives(data_root: Path, keep_zips: bool) -> None:
    split_archives = assemble_split_archives(data_root)
    archives = sorted(data_root.rglob("*.zip"))
    if not archives:
        print(f"no zip archives found under {data_root}, skipping extraction")
        return

    for archive in archives:
        print(f"extracting {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_root)
        if not keep_zips:
            archive.unlink()

    if not keep_zips:
        for part in split_archives:
            part.unlink()


def assemble_split_archives(data_root: Path) -> list[Path]:
    split_parts = sorted(data_root.rglob("*.zip.[0-9][0-9][0-9]"))
    if not split_parts:
        return []

    groups: dict[Path, list[Path]] = {}
    for part in split_parts:
        groups.setdefault(split_archive_base(part), []).append(part)

    for combined_path, parts in groups.items():
        expected_suffixes = [f"{i:03d}" for i in range(1, len(parts) + 1)]
        actual_suffixes = [part.suffix.lstrip(".") for part in parts]
        if actual_suffixes != expected_suffixes:
            raise SystemExit(
                f"split archive parts for {combined_path.name} are incomplete or out of order: "
                f"expected suffixes {expected_suffixes[0]}..{expected_suffixes[-1]}, got {actual_suffixes}"
            )

        if combined_path.exists():
            print(f"combined archive already exists: {combined_path}")
            continue

        print(f"assembling {combined_path} from {len(parts)} split archive parts")
        # The upstream files are byte-split zip parts, so I stream them
        # into one normal .zip before using Python's zipfile module. This
        # avoids shell-specific `cat` assumptions and keeps the recovery
        # path readable if one numbered part is missing.
        with open(combined_path, "wb") as combined:
            for part in parts:
                with open(part, "rb") as source:
                    shutil.copyfileobj(source, combined, length=1024 * 1024)

    return split_parts


def split_archive_base(part: Path) -> Path:
    return part.with_suffix("")


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
