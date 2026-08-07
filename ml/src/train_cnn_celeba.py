"""Trains the project CNN on CelebA-Spoof.

This is the first model-training step beyond NUAA. It deliberately keeps
the same small `SpoofCNN` architecture so the question stays narrow:
does the current CNN approach still work when the dataset includes many
more subjects and richer spoof media, before we spend effort on a larger
backbone or service integration?
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from celeba_spoof_dataset import CelebASpoofDataset, load_celeba_spoof_split, summarize_samples
from cnn_dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM
from model import SpoofCNN
from train_cnn import run_validation, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "celeba_spoof",
        help="directory containing CelebA-Spoof images and split annotation JSON files",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "celeba_spoof",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit-train", type=int, default=None, help="debug limit for smoke tests")
    parser.add_argument("--limit-val", type=int, default=None, help="debug limit for smoke tests")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; keep 0 on macOS/sandboxed runs where PyTorch shared memory workers can fail",
    )
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_celeba_spoof_split(args.data_root, args.train_split, args.limit_train)
    val_samples = load_celeba_spoof_split(args.data_root, args.val_split, args.limit_val)
    print(summarize_samples(args.train_split, train_samples))
    print(summarize_samples(args.val_split, val_samples))

    device = select_device()
    print(f"using device: {device}")

    train_ds = CelebASpoofDataset(train_samples, TRAIN_TRANSFORM)
    val_ds = CelebASpoofDataset(val_samples, EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpoofCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acer = float("inf")
    best_path = args.model_dir / "cnn_celeba_best.pt"

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
    print("run evaluate_cnn_celeba.py next for the held-out CelebA-Spoof test numbers")


if __name__ == "__main__":
    main()
