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
        help="keep downloaded zip files after successful extraction",
    )
    args = parser.parse_args()

    require_dataset_terms(args.i_accept_non_commercial_research_terms)
    args.data_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_google_drive_folder(args.google_drive_url, args.data_root)

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


def download_google_drive_folder(url: str, output_dir: Path) -> None:
    if importlib.util.find_spec("gdown") is None:
        raise SystemExit(
            "gdown is required for the Google Drive folder download. "
            "Install this repo's requirements first with `python -m pip install -r requirements.txt`."
        )

    # I call gdown through its CLI instead of importing private Python
    # APIs because the CLI is the stable interface users already know
    # from dataset download instructions. This also keeps authentication
    # and Drive confirmation handling inside gdown, where that brittle
    # host-specific logic belongs.
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        url,
        "-O",
        str(output_dir),
    ]
    print("downloading CelebA-Spoof from Google Drive")
    subprocess.run(cmd, check=True)


def extract_archives(data_root: Path, keep_zips: bool) -> None:
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
