# ml/src/model.py
"""A deliberately small CNN, not a pretrained backbone. The point of M2
is to see what a network trained from scratch, with no prior visual
knowledge, can learn about this specific texture/artifact problem, using
a pretrained ImageNet backbone here would answer a different question
(transfer learning performance) than the one this milestone is asking.
"""

import torch.nn as nn


class SpoofCNN(nn.Module):
    # Four conv blocks, channel width doubling each time, is a standard
    # small-CNN shape, not tuned further here, same reasoning as the LBP
    # baseline's untuned SVM hyperparameters: get a real number first,
    # tune once there's something to tune against.
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
        )

        # Global average pool instead of flattening straight into a
        # large FC layer, keeps the parameter count small and makes the
        # network agnostic to exact input size if that ever changes,
        # a flattened FC layer would break the moment IMAGE_SIZE changes.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Single logit, not two-class softmax, BCEWithLogitsLoss expects
        # this shape and it's the more direct fit for "live vs spoof"
        # than treating it as a general multi-class problem it isn't.
        self.classifier = nn.Linear(128, 1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(1)