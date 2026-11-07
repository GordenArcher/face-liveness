"""Downloads and prepares CelebA-Spoof into this project's data layout.

The official CelebA-Spoof distribution is hosted behind Google Drive
and Baidu Drive links rather than a plain versioned HTTP archive. That
means a downloader is useful, but it should not silently hide the
dataset terms or pretend the external host is as stable as a package
registry.

The steps are split into sibling modules so each can be read and
changed independently:

- `terms.py` requires the caller to acknowledge the dataset agreement
- `preflight.py` checks free space before expensive work starts
- `transfer/gdrive.py` drives `gdown` against the official Drive folder
- `archive/` assembles split parts and extracts the combined zip
- `progress.py` keeps long runs printing bounded progress
- `validate.py` confirms the training loader can find every split

`cli.py` wires them together in the order the dataset needs.
"""
