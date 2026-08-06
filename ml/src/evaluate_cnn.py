"""Evaluates the trained M2 CNN on the same held-out test split
evaluate.py used for M1, printed in the same format specifically so the
two are easy to put side by side.
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
    args = parser.parse_args()

    split_path = args.model_dir / "split.json"
    weights_path = args.model_dir / "cnn_best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"no trained weights at {weights_path}, run train_cnn.py first")

    device = select_device()

    test_ds = SplitDataset.from_split_file(split_path, "test", EVAL_TRANSFORM)
    print(f"evaluating on {len(test_ds)} held-out test images")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)

    print()
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"APCER:    {metrics['apcer']:.4f}  (spoof samples let through as live)")
    print(f"BPCER:    {metrics['bpcer']:.4f}  (live samples wrongly rejected as spoof)")
    print(f"ACER:     {metrics['acer']:.4f}  (average of the two, the headline number)")


if __name__ == "__main__":
    main()
