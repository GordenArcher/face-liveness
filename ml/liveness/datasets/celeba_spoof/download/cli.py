"""CLI orchestration for downloading and preparing CelebA-Spoof.

This module only wires the steps together in the order the dataset
needs. The actual checks and I/O live in sibling modules so each can be
read and changed independently.
"""

import argparse
from pathlib import Path

from liveness.datasets.celeba_spoof.download.archive.assemble import assemble_split_archives
from liveness.datasets.celeba_spoof.download.archive.extract import extract_archives
from liveness.datasets.celeba_spoof.download.preflight import assert_enough_free_space
from liveness.datasets.celeba_spoof.download.progress import Color, configure_color, log_status
from liveness.datasets.celeba_spoof.download.terms import require_dataset_terms
from liveness.datasets.celeba_spoof.download.transfer.gdrive import (
    OFFICIAL_GOOGLE_DRIVE_URL,
    download_google_drive_folder,
)
from liveness.datasets.celeba_spoof.download.validate import validate_prepared_dataset
from liveness.paths import DEFAULT_CELEBA_SPOOF_DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_CELEBA_SPOOF_DATA_ROOT,
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
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "where the assembled zip AND the extracted dataset are written; defaults to "
            "--data-root. Set this to a separate (e.g. external) volume when the split parts "
            "already fill most of --data-root's disk, since assembly + extraction together need "
            "roughly 2-3x the compressed dataset size in free space"
        ),
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
    # output_dir is resolved once, here, instead of threading an Optional[Path]
    # through every helper below. Assembly and extraction both write the bulk
    # of the data, so they both need to land wherever the caller actually has
    # room, not wherever the split parts happened to be downloaded.
    output_dir = args.output_dir if args.output_dir is not None else args.data_root
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_space_check:
        # data_root only needs headroom when something is still going to be
        # written there: a fresh download, or (when output_dir defaults to
        # data_root) the combined zip and extraction too. With
        # --skip-download and a separate --output-dir, data_root already
        # holds everything it needs to hold, so demanding min_free_gb free
        # there as well would block exactly the "parts fill the internal
        # disk, send the rest external" case --output-dir exists for.
        if not args.skip_download or output_dir == args.data_root:
            assert_enough_free_space(args.data_root, args.min_free_gb)
        # Checked separately because this is commonly a different physical
        # disk than data_root. A pass on data_root's free space says
        # nothing about the volume that receives the combined zip and
        # every extracted file.
        if output_dir != args.data_root:
            assert_enough_free_space(output_dir, args.min_free_gb)

    if not args.skip_download:
        download_google_drive_folder(
            args.google_drive_url,
            args.data_root,
            remaining_ok=not args.strict_folder_limit,
        )

    if args.assemble_only:
        assembly = assemble_split_archives(args.data_root, output_dir=output_dir)
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
            output_dir=output_dir,
        )

    if args.no_extract:
        log_status(
            "done",
            f"CelebA-Spoof archive parts are downloaded under {args.data_root}. "
            "Rerun without --no-extract on a volume with enough free space to assemble, extract, and validate.",
            color=Color.GREEN,
        )
    else:
        validate_prepared_dataset(output_dir)
        log_status("ready", f"CelebA-Spoof is ready at {output_dir}", color=Color.GREEN)


if __name__ == "__main__":
    main()
