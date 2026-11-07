"""Epoch validation shared by every CNN trainer.

The NUAA and CelebA-Spoof trainers select their best checkpoint by
validation ACER, not validation loss, so they need the same validation
pass. Keeping one implementation means the two training paths cannot
quietly start selecting checkpoints by different criteria.
"""

import numpy as np
import torch
from torch import nn

from liveness.metrics.pad import compute_metrics

LIVE_THRESHOLD = 0.5


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

            preds = (torch.sigmoid(logits) >= LIVE_THRESHOLD).float()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics
