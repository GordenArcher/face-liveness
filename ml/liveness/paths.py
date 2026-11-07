"""Filesystem defaults shared by every entry point.

The scripts used to derive `models/` and `data/` from their own
`__file__` location. That only worked while every module lived directly
in `ml/src/`. Centralizing the two roots here means a module can move
between capability folders without its default paths silently drifting
to the wrong directory.
"""

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = ML_ROOT / "models"
DEFAULT_NUAA_DATA_ROOT = ML_ROOT / "data" / "raw"
DEFAULT_CELEBA_SPOOF_DATA_ROOT = ML_ROOT / "data" / "celeba_spoof"
DEFAULT_CELEBA_SPOOF_MODEL_DIR = DEFAULT_MODEL_DIR / "celeba_spoof"
