"""K-fold cross-validation for the CNN across subject-disjoint folds.

train_cnn.py's single train/val/test split showed a real problem: with
only a handful of subjects in the validation set, val ACER is a noisy,
high-variance measurement, and picking "the epoch with the best val
ACER" out of many epochs is itself a form of overfitting to that noise.
The gap between the reported val ACER (0.0007) and the actual test ACER
(0.2216) in the first corrected run was the direct evidence of this.

Cross-validation doesn't remove the small-sample problem underneath,
NUAA only has a few dozen subjects total, but averaging ACER across k
independently trained folds is a genuinely more robust estimate than
trusting a single split's number, and the standard deviation across
folds is itself informative: small means the estimate is stable, large
is an honest signal it still shouldn't be trusted to one decimal place.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM, SplitDataset
from dataset import Sample, load_samples, train_val_split
from evaluate import compute_metrics
from kfold import make_folds, summarize_folds
from model import SpoofCNN
from train_cnn import run_validation, select_device


def _entries(samples: list[Sample]) -> list[dict]:
    return [{"path": str(s.path), "label": s.label} for s in samples]


def train_one_fold(train_samples, val_samples, device, epochs, batch_size, lr, num_workers) -> nn.Module:
    train_ds = SplitDataset(_entries(train_samples), TRAIN_TRANSFORM)
    val_ds = SplitDataset(_entries(val_samples), EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SpoofCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acer = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        _, val_metrics = run_validation(model, val_loader, device)
        if val_metrics["acer"] < best_acer:
            best_acer = val_metrics["acer"]
            # Cloned rather than referenced, state_dict() returns tensors
            # that keep training, saving a reference here would silently
            # end up holding the LAST epoch's weights, not the best one.
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate_on(model: nn.Module, samples: list[Sample], device, batch_size: int, num_workers: int) -> dict:
    ds = SplitDataset(_entries(samples), EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = (torch.sigmoid(model(images)) >= 0.5).float()
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; keep 0 on macOS/sandboxed runs where PyTorch shared memory workers can fail",
    )
    parser.add_argument(
        "--inner-val-size",
        type=float,
        default=0.15,
        help="fraction of each fold's training subjects held out for epoch selection",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"using device: {device}")

    samples = load_samples(args.data_root)
    folds = make_folds(samples, args.n_splits)
    print(f"{len(folds)} folds across {len({s.subject_id for s in samples})} subjects")
    print("this trains a fresh model per fold, expect roughly n_splits times a single training run")

    fold_metrics = []
    for i, (fold_train, fold_test) in enumerate(folds, start=1):
        print(f"\n--- fold {i}/{len(folds)} ---")
        print(
            f"train subjects: {len({s.subject_id for s in fold_train})}, "
            f"test subjects: {len({s.subject_id for s in fold_test})}"
        )

        # Carved out of this fold's training subjects only, for epoch
        # selection. The fold's test subjects, held in fold_test, are
        # never touched until the single evaluate_on call below.
        inner_train, inner_val = train_val_split(fold_train, val_size=args.inner_val_size, seed=42 + i)

        model = train_one_fold(
            inner_train,
            inner_val,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.num_workers,
        )
        metrics = evaluate_on(model, fold_test, device, args.batch_size, args.num_workers)
        print(
            f"fold {i} test ACER: {metrics['acer']:.4f}  "
            f"(APCER {metrics['apcer']:.4f}, BPCER {metrics['bpcer']:.4f})"
        )
        fold_metrics.append(metrics)

    summarize_folds(fold_metrics)


if __name__ == "__main__":
    main()
