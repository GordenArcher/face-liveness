"""Result of assembling CelebA-Spoof's split zip parts.

Kept as its own module so the extractor and the CLI can refer to the
same shape without importing the assembly implementation.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitArchiveAssembly:
    parts: list[Path]
    combined_archives: list[Path]
