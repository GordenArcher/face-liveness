"""CelebA-Spoof license acknowledgement gate.

The dataset is restricted to non-commercial research/education use, so
the download path must not proceed unless the caller explicitly accepts
the terms. Keeping the check separate makes it hard to accidentally drop
when the downloader is refactored.
"""


def require_dataset_terms(accepted: bool) -> None:
    if accepted:
        return

    raise SystemExit(
        "CelebA-Spoof is restricted to non-commercial research/education use. "
        "Read the official dataset agreement, then rerun with "
        "--i-accept-non-commercial-research-terms if you accept it."
    )
