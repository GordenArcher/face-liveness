"""Reusable liveness inference boundary for the trained PAD models.

Training and evaluation scripts are useful while measuring a model, but
the rest of the system needs a smaller contract: given one already
cropped face image, decide whether recognition is allowed to continue.
This module is that contract. It keeps the baseline and CNN loading
details behind one interface so a future gRPC service or Go gateway does
not need to know which files make up each model family.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import joblib
import numpy as np
import torch
from PIL import Image

from cnn_dataset import EVAL_TRANSFORM
from features import extract_lbp_histogram
from model import SpoofCNN
from train_cnn import select_device

LIVE_LABEL = "live"
SPOOF_LABEL = "spoof"
ModelKind = Literal["baseline", "cnn"]


@dataclass(frozen=True)
class LivenessResult:
    model: ModelKind
    label: str
    live_score: float
    spoof_score: float
    threshold: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


class LivenessPredictor(Protocol):
    def predict(self, image_path: Path) -> LivenessResult:
        """Returns a liveness gate decision for one already-cropped face image."""


class BaselineLivenessPredictor:
    def __init__(self, model_dir: Path, threshold: float = 0.5):
        self.model_dir = model_dir
        self.threshold = threshold
        self.model_path = model_dir / "lbp_svm.joblib"
        self.scaler_path = model_dir / "lbp_scaler.joblib"

        if not self.model_path.exists():
            raise FileNotFoundError(f"no baseline model at {self.model_path}, run train_baseline.py first")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"no baseline scaler at {self.scaler_path}, run train_baseline.py first")

        self.clf = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(self, image_path: Path) -> LivenessResult:
        features = extract_lbp_histogram(image_path).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        live_score = self._live_score(features_scaled)
        return _result("baseline", live_score, self.threshold)

    def _live_score(self, features_scaled: np.ndarray) -> float:
        # train_baseline.py fits SVC(probability=True), so the normal
        # path returns a calibrated-ish class probability for label 1
        # (live). The fallback exists because older local artifacts may
        # have been trained without probability support; in that case
        # decision_function still gives a monotonic margin we can squash
        # into a score-like value for the same gate contract.
        if hasattr(self.clf, "predict_proba"):
            class_index = list(self.clf.classes_).index(1)
            return float(self.clf.predict_proba(features_scaled)[0, class_index])

        margin = float(self.clf.decision_function(features_scaled)[0])
        return float(1.0 / (1.0 + np.exp(-margin)))


class CNNLivenessPredictor:
    def __init__(self, model_dir: Path, threshold: float = 0.5, device: torch.device | None = None):
        self.model_dir = model_dir
        self.threshold = threshold
        self.weights_path = model_dir / "cnn_best.pt"
        self.device = device or select_device()

        if not self.weights_path.exists():
            raise FileNotFoundError(f"no CNN weights at {self.weights_path}, run train_cnn.py first")

        self.model = SpoofCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_path: Path) -> LivenessResult:
        image = Image.open(image_path).convert("RGB")
        batch = EVAL_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            live_score = float(torch.sigmoid(self.model(batch))[0].cpu().item())

        return _result("cnn", live_score, self.threshold)


def load_predictor(model: ModelKind, model_dir: Path, threshold: float = 0.5) -> LivenessPredictor:
    if model == "baseline":
        return BaselineLivenessPredictor(model_dir, threshold)
    if model == "cnn":
        return CNNLivenessPredictor(model_dir, threshold)
    raise ValueError(f"unsupported liveness model {model!r}")


def _result(model: ModelKind, live_score: float, threshold: float) -> LivenessResult:
    passed = live_score >= threshold
    return LivenessResult(
        model=model,
        label=LIVE_LABEL if passed else SPOOF_LABEL,
        live_score=live_score,
        spoof_score=1.0 - live_score,
        threshold=threshold,
        passed=passed,
    )
