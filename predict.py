from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.har.data import CACHE_VERSION, HARDataset, build_or_load_cache
from src.har.model import SequenceTabularModel
from src.har.train_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test predictions for HAR.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="artifacts/cache")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--output", default="artifacts/submission.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-checkpoints", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def normalize_inputs(
    sequences: np.ndarray,
    features: np.ndarray,
    stats: dict[str, np.ndarray | torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    seq_mean = to_numpy(stats["seq_mean"])
    seq_std = to_numpy(stats["seq_std"])
    feat_mean = to_numpy(stats["feat_mean"])
    feat_std = to_numpy(stats["feat_std"])
    norm_sequences = (sequences - seq_mean) / seq_std
    norm_features = (features - feat_mean) / feat_std
    return norm_sequences.astype(np.float32), norm_features.astype(np.float32)


def to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def relative_label(checkpoint_dir: Path, checkpoint_path: Path) -> str:
    return str(checkpoint_path.relative_to(checkpoint_dir))


def seed_group_label(checkpoint_dir: Path, checkpoint_path: Path) -> str:
    relative_parent = checkpoint_path.parent.relative_to(checkpoint_dir)
    if str(relative_parent) == ".":
        return "<root>"
    return str(relative_parent)


def print_checkpoint_summary(checkpoint_dir: Path, checkpoint_paths: list[Path]) -> None:
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for checkpoint_path in checkpoint_paths:
        grouped_paths[seed_group_label(checkpoint_dir, checkpoint_path)].append(checkpoint_path)

    print(f"Found {len(checkpoint_paths)} checkpoint(s) under {checkpoint_dir}")
    print("Checkpoint summary:")
    for group_name in sorted(grouped_paths):
        group_paths = sorted(grouped_paths[group_name])
        fold_names = ", ".join(path.stem for path in group_paths)
        print(f"- {group_name}: {len(group_paths)} checkpoint(s) [{fold_names}]")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.rglob("fold_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    print_checkpoint_summary(checkpoint_dir, checkpoint_paths)
    if args.expected_checkpoints is not None and len(checkpoint_paths) != args.expected_checkpoints:
        raise RuntimeError(
            f"Expected {args.expected_checkpoints} checkpoint(s) under {checkpoint_dir}, "
            f"but found {len(checkpoint_paths)}."
        )
    if args.summary_only:
        return

    test_cache = build_or_load_cache(args.data_dir, "test", args.cache_dir)
    ensemble_logits = None
    loaded_count = 0

    for checkpoint_path in checkpoint_paths:
        bundle = load_checkpoint(checkpoint_path, device)
        config = bundle["config"]
        stats = bundle["stats"]
        checkpoint_version = config.get("cache_version")
        feature_dim = int(to_numpy(stats["feat_mean"]).shape[-1])
        version_mismatch = checkpoint_version not in (None, CACHE_VERSION)
        feature_mismatch = feature_dim != int(test_cache.features.shape[1])
        if version_mismatch or feature_mismatch:
            print(
                f"Skipped {checkpoint_path.name} because "
                f"cache_version={checkpoint_version} and feature_dim={feature_dim}; "
                f"expected cache_version={CACHE_VERSION} and feature_dim={test_cache.features.shape[1]}"
            )
            continue

        sequences, features = normalize_inputs(test_cache.sequences, test_cache.features, stats)
        if features.shape[1] != config["feature_dim"]:
            print(f"Skipped {checkpoint_path.name} because feature_dim={config['feature_dim']} but cache has {features.shape[1]}")
            continue
        loader = DataLoader(
            HARDataset(sequences, features),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        model = SequenceTabularModel(
            sequence_dim=config["sequence_dim"],
            feature_dim=config["feature_dim"],
            num_classes=config["num_classes"],
        ).to(device)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        logits_buffer: list[np.ndarray] = []
        with torch.no_grad():
            for batch_sequences, batch_features in loader:
                batch_sequences = batch_sequences.to(device)
                batch_features = batch_features.to(device)
                logits = model(batch_sequences, batch_features)
                logits_buffer.append(logits.cpu().numpy())

        fold_logits = np.concatenate(logits_buffer, axis=0)
        ensemble_logits = fold_logits if ensemble_logits is None else ensemble_logits + fold_logits
        loaded_count += 1
        print(f"Loaded {relative_label(checkpoint_dir, checkpoint_path)}")

    if loaded_count == 0:
        raise RuntimeError("No compatible checkpoints were loaded. Retrain with the current code version or point to a matching checkpoint directory.")
    if args.expected_checkpoints is not None and loaded_count != args.expected_checkpoints:
        raise RuntimeError(
            f"Expected to load {args.expected_checkpoints} compatible checkpoint(s), "
            f"but loaded {loaded_count}."
        )

    print(f"Loaded {loaded_count} compatible checkpoint(s) for ensemble.")

    # 儲存 Neural Network Logits 供後續 Ensemble 使用
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path.parent / "nn_logits.npy", ensemble_logits)
    print(f"Saved neural network logits to {output_path.parent / 'nn_logits.npy'}")
    
    predictions = ensemble_logits.argmax(axis=1)
    submission = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    prediction_map = {int(file_id): int(label) for file_id, label in zip(test_cache.file_ids, predictions)}
    submission["Label"] = submission["Id"].map(prediction_map).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
