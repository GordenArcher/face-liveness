"""Google Drive folder download for CelebA-Spoof via gdown.

The official distribution is a Drive folder of split archive parts, not
a plain HTTP archive, so a generic download helper does not work here.
The CLI is used instead of gdown's private Python APIs because the CLI
is the stable interface documented by the dataset instructions.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

from liveness.datasets.celeba_spoof.download.preflight import is_no_space_left
from liveness.datasets.celeba_spoof.download.progress import Color, log_status

OFFICIAL_GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing"


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
