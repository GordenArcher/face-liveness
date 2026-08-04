"""Trains the M1 classical baseline: LBP histograms + SVM.

Saves the exact train/val/test split to disk rather than letting
evaluate.py recompute its own split independently, if the splitting
logic ever changes between when this runs and when evaluate.py runs,
recomputing could silently produce a different test set than what was
actually held out during training, which would make the reported
numbers meaningless without anyone noticing.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_samples, train_val_test_split, labels_array
from features import extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="directory containing ClientFace/ and ImposterFace/",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
    )
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading samples from {args.data_root}")
    samples = load_samples(args.data_root)
    print(f"found {len(samples)} images")

    train, val, test = train_val_test_split(samples)
    print(f"split: {len(train)} train, {len(val)} val, {len(test)} test")

    # Written before training even starts, if feature extraction or
    # training crashes partway through, the split itself is still
    # recoverable for a retry rather than needing to be recomputed
    # (and potentially landing differently) on the next attempt.
    # Labels stored alongside paths, not re-derived from folder structure
    # in evaluate.py, that would mean duplicating the ClientFace/
    # ImposterFace parsing logic in two places and risking them drifting
    # apart, this way evaluate.py just reads what training already knows.
    split_record = {
        "train": [{"path": str(s.path), "label": s.label} for s in train],
        "val": [{"path": str(s.path), "label": s.label} for s in val],
        "test": [{"path": str(s.path), "label": s.label} for s in test],
    }
    with open(args.model_dir / "split.json", "w") as f:
        json.dump(split_record, f, indent=2)

    print("extracting LBP features (train)")
    X_train = extract_features(train)
    y_train = labels_array(train)

    print("extracting LBP features (val)")
    X_val = extract_features(val)
    y_val = labels_array(val)

    # LBP histograms are already normalized per-image (see features.py),
    # this scaler normalizes per-feature across the dataset instead, SVMs
    # with an RBF kernel are sensitive to feature scale, skipping this
    # step is a common way to quietly get a worse model without any
    # error to point at why.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("training SVM")
    # RBF kernel as the starting point, not tuned via grid search yet,
    # that tuning is worth doing once this baseline number exists to
    # compare against, not before.
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    clf.fit(X_train_scaled, y_train)

    train_acc = clf.score(X_train_scaled, y_train)
    val_acc = clf.score(X_val_scaled, y_val)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"val accuracy:   {val_acc:.4f}")

    joblib.dump(clf, args.model_dir / "lbp_svm.joblib")
    joblib.dump(scaler, args.model_dir / "lbp_scaler.joblib")
    print(f"saved model + scaler to {args.model_dir}")
    print("run evaluate.py next for the real held-out numbers (APCER/BPCER/ACER), not the val accuracy above")


if __name__ == "__main__":
    main()