"""Mean and standard deviation across cross-validation folds.

Shared by both cross-validation entry points so the two summaries are
computed the same way. Two copies could report the same fold metrics
with different rounding or a different std convention without anyone
noticing.
"""

import numpy as np


def summarize_folds(fold_metrics: list[dict]) -> None:
    acers = np.array([m["acer"] for m in fold_metrics])
    apcers = np.array([m["apcer"] for m in fold_metrics])
    bpcers = np.array([m["bpcer"] for m in fold_metrics])

    print("\n=== cross-validation summary ===")
    print(f"ACER:  {acers.mean():.4f} +/- {acers.std():.4f}")
    print(f"APCER: {apcers.mean():.4f} +/- {apcers.std():.4f}")
    print(f"BPCER: {bpcers.mean():.4f} +/- {bpcers.std():.4f}")
