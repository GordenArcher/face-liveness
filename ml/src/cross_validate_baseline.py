"""K-fold cross-validation for the LBP+SVM baseline.

Uses the same kfold.make_folds as cross_validate_cnn.py. GroupKFold's
fold assignment is deterministic given the same input order and
n_splits, and load_samples walks the filesystem in the same order every
run for a fixed dataset on disk, so both scripts see identical folds
without needing to explicitly share state, this is what makes the two
models' cross-validation summaries a fair, apples-to-apples comparison
rather than each being evaluated against different data.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_samples
from evaluate import compute_metrics
from features import extract_features
from kfold import make_folds, summarize_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    samples = load_samples(args.data_root)
    folds = make_folds(samples, args.n_splits)
    print(f"{len(folds)} folds across {len({s.subject_id for s in samples})} subjects")

    fold_metrics = []
    for i, (train_samples, test_samples) in enumerate(folds, start=1):
        print(f"\n--- fold {i}/{len(folds)} ---")

        X_train = extract_features(train_samples)
        y_train = np.array([s.label for s in train_samples])
        X_test = extract_features(test_samples)
        y_test = np.array([s.label for s in test_samples])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        metrics = compute_metrics(y_test, y_pred)
        print(
            f"fold {i} test ACER: {metrics['acer']:.4f}  "
            f"(APCER {metrics['apcer']:.4f}, BPCER {metrics['bpcer']:.4f})"
        )
        fold_metrics.append(metrics)

    summarize_folds(fold_metrics)


if __name__ == "__main__":
    main()