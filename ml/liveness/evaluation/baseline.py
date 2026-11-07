"""Evaluates the trained LBP+SVM baseline on the held-out test split
saved by the baseline trainer.

Reports APCER/BPCER/ACER, not just accuracy. Accuracy alone hides which
failure mode is actually happening: a model that always predicts "live"
can post a deceptively high accuracy if the test set happens to be
mostly live samples, while its APCER (letting spoofs through) would be
100%. These three metrics are the standard the presentation-attack-
detection literature actually reports in, worth using them by name.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from liveness.datasets.nuaa.samples import Sample
from liveness.features.lbp import extract_features
from liveness.metrics.pad import compute_metrics
from liveness.paths import DEFAULT_MODEL_DIR


def load_test_split(model_dir: Path) -> list[Sample]:
    split_path = model_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"no split.json found at {split_path}, run `python -m liveness.training.baseline` first, "
            "this file is what pins down the exact held-out test set training used"
        )
    with open(split_path) as f:
        record = json.load(f)
    return [
        Sample(path=Path(e["path"]), label=e["label"], subject_id=Path(e["path"]).parent.name)
        for e in record["test"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    args = parser.parse_args()

    test_samples = load_test_split(args.model_dir)
    print(f"evaluating on {len(test_samples)} held-out test images")

    clf = joblib.load(args.model_dir / "lbp_svm.joblib")
    scaler = joblib.load(args.model_dir / "lbp_scaler.joblib")

    X_test = extract_features(test_samples)
    X_test_scaled = scaler.transform(X_test)
    y_test = np.array([s.label for s in test_samples])

    y_pred = clf.predict(X_test_scaled)

    metrics = compute_metrics(y_test, y_pred)

    print()
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"APCER:    {metrics['apcer']:.4f}  (spoof samples let through as live)")
    print(f"BPCER:    {metrics['bpcer']:.4f}  (live samples wrongly rejected as spoof)")
    print(f"ACER:     {metrics['acer']:.4f}  (average of the two, the headline number)")


if __name__ == "__main__":
    main()