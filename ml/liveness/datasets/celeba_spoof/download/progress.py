"""Progress and colored status logging for long dataset preparation runs.

CelebA-Spoof downloads and extracts for hours, so silence is ambiguous:
the job may be healthy, the external disk may be asleep, or the volume
may have stopped accepting writes. These helpers keep each phase
printing at a bounded interval without flooding a piped log.
"""

import os
import sys
import time

PROGRESS_INTERVAL_SECONDS = 10.0
USE_COLOR = False


class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


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
