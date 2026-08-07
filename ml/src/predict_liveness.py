"""CLI entry point for the liveness gate.

The training scripts answer "how good is this model on a dataset?".
This script answers the integration question the rest of the project
actually needs: "if I pass this face crop through the liveness gate,
does recognition continue or stop?"
"""

import argparse
import json
from pathlib import Path

from liveness import load_predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="path to an already-cropped face image")
    parser.add_argument(
        "--model",
        choices=["baseline", "cnn"],
        default="cnn",
        help="trained liveness model to use",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum live score required to allow recognition to continue",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"no image found at {args.image}")

    predictor = load_predictor(args.model, args.model_dir, args.threshold)
    result = predictor.predict(args.image)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
