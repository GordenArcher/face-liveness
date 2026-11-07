"""Evaluates a CelebA-Spoof-trained CNN on a CelebA-Spoof split."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from liveness.datasets.celeba_spoof.samples import load_celeba_spoof_split, summarize_samples
from liveness.datasets.celeba_spoof.torch_dataset import CelebASpoofDataset
from liveness.datasets.transforms import EVAL_TRANSFORM
from liveness.device import select_device
from liveness.metrics.pad import compute_metrics
from liveness.models.spoof_cnn import SpoofCNN
from liveness.paths import DEFAULT_CELEBA_SPOOF_DATA_ROOT, DEFAULT_CELEBA_SPOOF_MODEL_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_CELEBA_SPOOF_DATA_ROOT,
        help="directory containing CelebA-Spoof images and split annotation JSON files",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_CELEBA_SPOOF_MODEL_DIR,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="debug limit for smoke tests")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; keep 0 on macOS/sandboxed runs where PyTorch shared memory workers can fail",
    )
    args = parser.parse_args()

    weights_path = args.model_dir / "cnn_celeba_best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"no CelebA-Spoof CNN weights at {weights_path}, run `python -m liveness.training.celeba_spoof` first")

    samples = load_celeba_spoof_split(args.data_root, args.split, args.limit)
    print(summarize_samples(args.split, samples))

    device = select_device()
    ds = CelebASpoofDataset(samples, EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) >= args.threshold).float()
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)

    print()
    print(f"threshold: {args.threshold:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"APCER:    {metrics['apcer']:.4f}  (spoof samples let through as live)")
    print(f"BPCER:    {metrics['bpcer']:.4f}  (live samples wrongly rejected as spoof)")
    print(f"ACER:     {metrics['acer']:.4f}  (average of the two, the headline number)")


if __name__ == "__main__":
    main()
