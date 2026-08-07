"""Sweeps CNN live-score thresholds on the saved validation and test split.

The CNN emits a live probability-like score and the old evaluation path
used a fixed 0.5 cutoff. That is a decent default for proving the model
can learn, but it is not a security policy. A liveness gate usually
needs to trade off APCER (spoofs accepted) against BPCER (real users
rejected), and that tradeoff is controlled by the live threshold.

This script intentionally selects a threshold on the validation split
first, then reports what that same threshold does on the held-out test
split. Looking at test performance for every threshold is still useful
for diagnosis, but choosing the threshold directly on test data would
overfit the test set and make the reported number too optimistic.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_dataset import EVAL_TRANSFORM, SplitDataset
from evaluate import compute_metrics
from model import SpoofCNN
from train_cnn import select_device


def collect_scores(model: SpoofCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    scores = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_scores = torch.sigmoid(model(images)).cpu().numpy()
            labels.append(batch_labels.numpy())
            scores.append(batch_scores)

    return np.concatenate(labels), np.concatenate(scores)


def metrics_at_threshold(labels: np.ndarray, live_scores: np.ndarray, threshold: float) -> dict:
    preds = (live_scores >= threshold).astype(float)
    return compute_metrics(labels, preds)


def format_metrics(metrics: dict) -> str:
    return (
        f"accuracy {metrics['accuracy']:.4f}  "
        f"APCER {metrics['apcer']:.4f}  "
        f"BPCER {metrics['bpcer']:.4f}  "
        f"ACER {metrics['acer']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; keep 0 on macOS/sandboxed runs where PyTorch shared memory workers can fail",
    )
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--max-apcer",
        type=float,
        default=None,
        help="optional validation APCER ceiling; picks the lowest validation ACER among thresholds that satisfy it",
    )
    args = parser.parse_args()

    split_path = args.model_dir / "split.json"
    weights_path = args.model_dir / "cnn_best.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"no split.json at {split_path}, run train_baseline.py first")
    if not weights_path.exists():
        raise FileNotFoundError(f"no CNN weights at {weights_path}, run train_cnn.py first")

    device = select_device()
    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    val_ds = SplitDataset.from_split_file(split_path, "val", EVAL_TRANSFORM)
    test_ds = SplitDataset.from_split_file(split_path, "test", EVAL_TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    val_labels, val_scores = collect_scores(model, val_loader, device)
    test_labels, test_scores = collect_scores(model, test_loader, device)

    thresholds = np.arange(args.start, args.stop + args.step / 2, args.step)
    rows = []
    for threshold in thresholds:
        val_metrics = metrics_at_threshold(val_labels, val_scores, threshold)
        test_metrics = metrics_at_threshold(test_labels, test_scores, threshold)
        rows.append((float(threshold), val_metrics, test_metrics))

    candidates = rows
    if args.max_apcer is not None:
        candidates = [row for row in rows if row[1]["apcer"] <= args.max_apcer]
        if not candidates:
            raise ValueError(
                f"no threshold between {args.start} and {args.stop} keeps validation APCER <= {args.max_apcer:.4f}"
            )

    # Threshold selection is based only on validation metrics. Test
    # metrics are printed for the selected threshold after the fact so
    # we can estimate how the chosen gate behaves on held-out data.
    selected = min(candidates, key=lambda row: (row[1]["acer"], row[1]["bpcer"], row[0]))

    print(f"validation images: {len(val_ds)}, test images: {len(test_ds)}")
    print(f"selected threshold: {selected[0]:.2f}")
    if args.max_apcer is not None:
        print(f"selection constraint: validation APCER <= {args.max_apcer:.4f}")
    print(f"validation @ selected: {format_metrics(selected[1])}")
    print(f"test       @ selected: {format_metrics(selected[2])}")
    print("note: NUAA's validation split is small, treat this as threshold diagnostics, not a production policy")

    print("\nthreshold sweep")
    print("threshold | validation                         | test")
    for threshold, val_metrics, test_metrics in rows:
        print(f"{threshold:9.2f} | {format_metrics(val_metrics)} | {format_metrics(test_metrics)}")


if __name__ == "__main__":
    main()
