"""Presentation attack detection metrics: APCER, BPCER, ACER.

Every trainer and evaluator reports these three numbers, so this is the
one implementation. Accuracy is included only because it is useful for
sanity checks; it is not the headline metric because a model that always
predicts "live" can post a high accuracy while letting every spoof
through.
"""

import numpy as np

LIVE_LABEL = 1
SPOOF_LABEL = 0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    live_mask = y_true == LIVE_LABEL
    spoof_mask = y_true == SPOOF_LABEL

    # APCER: spoof samples wrongly classified as live, this is the
    # dangerous failure mode for a security-relevant classifier, a spoof
    # getting through.
    apcer = np.mean(y_pred[spoof_mask] == LIVE_LABEL) if spoof_mask.any() else float("nan")

    # BPCER: live samples wrongly classified as spoof, the annoying but
    # safe failure mode, a real user gets rejected.
    bpcer = np.mean(y_pred[live_mask] == SPOOF_LABEL) if live_mask.any() else float("nan")

    acer = (apcer + bpcer) / 2
    accuracy = np.mean(y_true == y_pred)

    return {"apcer": apcer, "bpcer": bpcer, "acer": acer, "accuracy": accuracy}
