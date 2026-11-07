"""The CNN must keep its single-logit, batch-preserving contract.

Training uses BCEWithLogitsLoss, which expects one logit per image, not
a two-class softmax. A forward-pass shape regression would surface much
later as a confusing loss error.
"""

import unittest

import torch

from liveness.models.spoof_cnn import SpoofCNN


class SpoofCNNTest(unittest.TestCase):
    def test_forward_returns_one_logit_per_image(self):
        model = SpoofCNN()
        batch = torch.randn(2, 3, 128, 128)

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())
