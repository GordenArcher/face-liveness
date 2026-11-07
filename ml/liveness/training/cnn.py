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

import torch
from torch import nn
from torch.utils.data import DataLoader

from liveness.datasets.nuaa.torch_dataset import SplitDataset
from liveness.datasets.transforms import EVAL_TRANSFORM, TRAIN_TRANSFORM
from liveness.device import select_device
from liveness.models.spoof_cnn import SpoofCNN
from liveness.paths import DEFAULT_MODEL_DIR
from liveness.training.validation import run_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
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
            f"no split.json at {split_path}, run `python -m liveness.training.baseline` first, "
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
    print("run `python -m liveness.evaluation.cnn` next for the held-out test numbers")


if __name__ == "__main__":
    main()
