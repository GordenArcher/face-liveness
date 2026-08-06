"""Trains the M2 CNN on the exact split M1 used (see split.json), so the
eventual M1-vs-M2 comparison is apples to apples.

Model selection is by validation ACER, not validation loss. Loss is what
the optimizer minimizes, but it's not the number this project actually
reports, picking the checkpoint with the lowest loss instead of the
lowest ACER could hand back a model that's numerically well-fit but not
actually the best one by the metric that matters here.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM, SplitDataset
from evaluate import compute_metrics
from model import SpoofCNN


def select_device() -> torch.device:
    # mps first, this is routinely run on Apple Silicon, falling back to
    # cpu on an M-series Mac would be a silent multi-x slowdown nobody
    # would notice the reason for.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_validation(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; keep 0 on macOS/sandboxed runs where PyTorch shared memory workers can fail",
    )
    args = parser.parse_args()

    split_path = args.model_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"no split.json at {split_path}, run train_baseline.py first, "
            "M2 trains on the same split M1 used, it doesn't compute its own"
        )

    device = select_device()
    print(f"using device: {device}")

    train_ds = SplitDataset.from_split_file(split_path, "train", TRAIN_TRANSFORM)
    val_ds = SplitDataset.from_split_file(split_path, "val", EVAL_TRANSFORM)
    print(f"train: {len(train_ds)} images, val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpoofCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acer = float("inf")
    best_path = args.model_dir / "cnn_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_metrics = run_validation(model, val_loader, device)

        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_acc {val_metrics['accuracy']:.4f}  val_acer {val_metrics['acer']:.4f}"
        )

        if val_metrics["acer"] < best_acer:
            best_acer = val_metrics["acer"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best val ACER {best_acer:.4f}, saved to {best_path}")

    print(f"training done, best val ACER: {best_acer:.4f}")
    print("run evaluate_cnn.py next for the held-out test numbers")


if __name__ == "__main__":
    main()
