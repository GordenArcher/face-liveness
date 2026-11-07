"""Post-extraction check that the training loader can find every split.

The downloader can finish successfully and still leave an unusable
directory if the upstream layout changed. Running the same
`find_annotation_file` the trainer uses turns that into an immediate,
specific error instead of a failure on the first training batch.
"""

from pathlib import Path

from liveness.datasets.celeba_spoof.samples import find_annotation_file


def validate_prepared_dataset(output_dir: Path) -> None:
    missing = []
    for split in ("train", "val", "test"):
        try:
            annotation_path = find_annotation_file(output_dir, split)
            print(f"found {split} annotations: {annotation_path}")
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        raise SystemExit(
            "CelebA-Spoof download/extraction finished, but the expected split annotations were not found:\n"
            + "\n".join(f"- {message}" for message in missing)
        )
