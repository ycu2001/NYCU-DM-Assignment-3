import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.har.data import build_or_load_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble LightGBM and CatBoost probabilities."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Path to the data directory"
    )
    parser.add_argument(
        "--lgb-probs",
        default="artifacts/lgb_test_probs.npy",
        help="Path to the LightGBM probabilities .npy file",
    )
    parser.add_argument(
        "--cat-probs",
        default="artifacts/catboost_test_probs.npy",
        help="Path to the CatBoost probabilities .npy file",
    )
    parser.add_argument(
        "--lgb-weight",
        type=float,
        default=0.5,
        help="Weight for LightGBM predictions (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/submission_lgb_catboost.csv",
        help="Path to save the final ensemble submission CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lgb_probs_path = Path(args.lgb_probs)
    cat_probs_path = Path(args.cat_probs)
    lgb_weight = args.lgb_weight
    cat_weight = 1.0 - lgb_weight

    if not 0.0 <= lgb_weight <= 1.0:
        raise ValueError("--lgb-weight must be between 0.0 and 1.0.")

    print(
        f"Blending models (LightGBM weight: {lgb_weight:.2f}, CatBoost weight: {cat_weight:.2f})..."
    )

    lgb_probs = None
    cat_probs = None

    if lgb_weight > 0.0:
        if not lgb_probs_path.exists():
            print(f"Error: LightGBM probabilities file not found at {lgb_probs_path}")
            return
        lgb_probs = np.load(lgb_probs_path)

    if cat_weight > 0.0:
        if not cat_probs_path.exists():
            print(f"Error: CatBoost probabilities file not found at {cat_probs_path}")
            return
        cat_probs = np.load(cat_probs_path)

    if lgb_probs is None and cat_probs is None:
        print("Error: both model weights are zero.")
        return

    if lgb_probs is None:
        final_probs = cat_probs
    elif cat_probs is None:
        final_probs = lgb_probs
    else:
        if lgb_probs.shape != cat_probs.shape:
            raise ValueError(
                f"Probability shape mismatch: LightGBM {lgb_probs.shape} vs CatBoost {cat_probs.shape}"
            )
        final_probs = (lgb_weight * lgb_probs) + (cat_weight * cat_probs)

    final_preds = np.argmax(final_probs, axis=1)

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
