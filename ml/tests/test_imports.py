"""Every module in the grouped package must still import.

This is the regression test for the reorganisation itself: moving a
module between capability folders is only safe if every absolute import
path was updated with it.
"""

import importlib
import unittest

MODULES = [
    "liveness.paths",
    "liveness.device",
    "liveness.datasets.transforms",
    "liveness.datasets.nuaa.samples",
    "liveness.datasets.nuaa.split",
    "liveness.datasets.nuaa.kfold",
    "liveness.datasets.nuaa.torch_dataset",
    "liveness.datasets.celeba_spoof.samples",
    "liveness.datasets.celeba_spoof.torch_dataset",
    "liveness.datasets.celeba_spoof.download.cli",
    "liveness.datasets.celeba_spoof.download.archive.model",
    "liveness.datasets.celeba_spoof.download.archive.assemble",
    "liveness.datasets.celeba_spoof.download.archive.extract",
    "liveness.datasets.celeba_spoof.download.transfer.gdrive",
    "liveness.datasets.celeba_spoof.download.preflight",
    "liveness.datasets.celeba_spoof.download.progress",
    "liveness.datasets.celeba_spoof.download.terms",
    "liveness.datasets.celeba_spoof.download.validate",
    "liveness.features.lbp",
    "liveness.models.spoof_cnn",
    "liveness.metrics.pad",
    "liveness.metrics.folds",
    "liveness.training.validation",
    "liveness.training.baseline",
    "liveness.training.cnn",
    "liveness.training.cross_validation.baseline",
    "liveness.training.cross_validation.cnn",
    "liveness.training.celeba_spoof",
    "liveness.evaluation.baseline",
    "liveness.evaluation.cnn",
    "liveness.evaluation.celeba_spoof",
    "liveness.evaluation.threshold_sweep",
    "liveness.inference.predictor",
    "liveness.inference.cli",
]


class ModuleImportsTest(unittest.TestCase):
    def test_every_module_imports(self):
        for name in MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)
