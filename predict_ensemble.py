import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.har.data import build_or_load_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble Neural Network logits and LightGBM probabilities."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Path to the data directory"
    )
    parser.add_argument(
        "--nn-logits",
        default="artifacts/nn_logits.npy",
        help="Path to the Neural Network logits .npy file",
    )
    parser.add_argument(
        "--lgb-probs",
        default="artifacts/lgb_test_probs.npy",
        help="Path to the LightGBM probabilities .npy file",
    )
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.6,
        help="Weight for Neural Network predictions (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/submission_ensemble.csv",
        help="Path to save the final ensemble submission CSV",
    )
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def main():
    args = parse_args()

    nn_logits_path = Path(args.nn_logits)
    lgb_probs_path = Path(args.lgb_probs)

    # Perform weighted blending
    nn_weight = args.nn_weight
    lgb_weight = 1.0 - nn_weight
    print(f"Blending models (NN weight: {nn_weight:.2f}, LightGBM weight: {lgb_weight:.2f})...")

    nn_probs = None
    lgb_probs = None

    if nn_weight > 0.0:
        if not nn_logits_path.exists():
            print(f"Error: NN logits file not found at {nn_logits_path}")
            return
        nn_logits = np.load(nn_logits_path)
        nn_probs = softmax(nn_logits)

    if lgb_weight > 0.0:
        if not lgb_probs_path.exists():
            print(f"Error: LightGBM probabilities file not found at {lgb_probs_path}")
            return
        lgb_probs = np.load(lgb_probs_path)

    if nn_probs is None and lgb_probs is None:
        print("Error: both model weights are zero.")
        return

    if nn_probs is None:
        final_probs = lgb_probs
    elif lgb_probs is None:
        final_probs = nn_probs
    else:
        final_probs = (nn_weight * nn_probs) + (lgb_weight * lgb_probs)

    final_preds = np.argmax(final_probs, axis=1)

    # Load test cache to map file IDs
    test_cache = build_or_load_cache(data_dir=args.data_dir, split="test")
    
    submission = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    prediction_map = {
        int(file_id): int(label)
        for file_id, label in zip(test_cache.file_ids, final_preds)
    }
    submission["Label"] = submission["Id"].map(prediction_map).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Ensemble submission successfully saved to {output_path}")


if __name__ == "__main__":
    main()
