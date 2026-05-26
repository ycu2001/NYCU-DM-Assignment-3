from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.har.data import CACHE_VERSION, HARDataset, build_or_load_cache
from src.har.model import SequenceTabularModel
from src.har.train_utils import (
    FocalLoss,
    ModelEMA,
    compute_class_weights,
    compute_effective_number_weights,
    macro_f1_score,
    make_stratified_group_folds,
    select_device,
    set_seed,
    summarize_label_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HAR classifier.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="artifacts/cache")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--class-weighting", choices=["none", "inverse", "effective"], default="inverse")
    parser.add_argument("--effective-beta", type=float, default=0.999)
    parser.add_argument("--sampler", choices=["weighted", "shuffle"], default="weighted")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--max-folds", type=int, default=None, help="Train only the first N folds for quick experiments.")
    return parser.parse_args()


def normalize_split(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    seq_mean = train_sequences.mean(axis=(0, 1), keepdims=True)
    seq_std = train_sequences.std(axis=(0, 1), keepdims=True)
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)

    feat_mean = train_features.mean(axis=0, keepdims=True)
    feat_std = train_features.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    normalized_train_sequences = (train_sequences - seq_mean) / seq_std
    normalized_val_sequences = (val_sequences - seq_mean) / seq_std
    normalized_train_features = (train_features - feat_mean) / feat_std
    normalized_val_features = (val_features - feat_mean) / feat_std

    stats = {
        "seq_mean": seq_mean.astype(np.float32),
        "seq_std": seq_std.astype(np.float32),
        "feat_mean": feat_mean.astype(np.float32),
        "feat_std": feat_std.astype(np.float32),
    }
    return (
        normalized_train_sequences.astype(np.float32),
        normalized_val_sequences.astype(np.float32),
        normalized_train_features.astype(np.float32),
        normalized_val_features.astype(np.float32),
        stats,
    )


def create_loader(
    sequences: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    workers: int,
    sampler_mode: str,
    training: bool,
) -> DataLoader:
    dataset = HARDataset(sequences, features, labels)
    if training and sampler_mode == "weighted":
        class_weights = compute_class_weights(labels)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=workers)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ema: ModelEMA | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in loader:
        sequences, features, targets = batch
        sequences = sequences.to(device)
        features = features.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            logits = model(sequences, features)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

        total_loss += float(loss.item()) * targets.size(0)
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, y_true, y_pred


def save_checkpoint(
    output_dir: Path,
    fold_id: int,
    model: nn.Module,
    stats: dict[str, np.ndarray],
    config: dict,
    val_f1: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized_stats = {name: torch.from_numpy(value.astype(np.float32)) for name, value in stats.items()}
    payload = {
        "model_state_dict": model.state_dict(),
        "stats": serialized_stats,
        "config": config,
        "val_f1": float(val_f1),
    }
    torch.save(payload, output_dir / f"fold_{fold_id}.pt")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    train_cache = build_or_load_cache(args.data_dir, "train", args.cache_dir)
    folds = make_stratified_group_folds(
        labels=train_cache.labels,
        groups=train_cache.user_ids,
        n_splits=args.folds,
        seed=args.seed,
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    oof_predictions = np.full(train_cache.labels.shape[0], -1, dtype=np.int64)
    print(f"Using device: {device}")
    print(f"Training samples: {train_cache.labels.shape[0]}")
    print(f"Label distribution: {summarize_label_distribution(train_cache.labels)}")

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\nFold {fold_id + 1}/{len(folds)}")
        train_sequences = train_cache.sequences[train_idx]
        val_sequences = train_cache.sequences[val_idx]
        train_features = train_cache.features[train_idx]
        val_features = train_cache.features[val_idx]
        train_labels = train_cache.labels[train_idx]
        val_labels = train_cache.labels[val_idx]

        (
            train_sequences,
            val_sequences,
            train_features,
            val_features,
            stats,
        ) = normalize_split(train_sequences, val_sequences, train_features, val_features)

        train_loader = create_loader(
            train_sequences,
            train_features,
            train_labels,
            batch_size=args.batch_size,
            workers=args.workers,
            sampler_mode=args.sampler,
            training=True,
        )
        val_loader = create_loader(
            val_sequences,
            val_features,
            val_labels,
            batch_size=args.batch_size,
            workers=args.workers,
            sampler_mode="shuffle",
            training=False,
        )

        model = SequenceTabularModel(
            sequence_dim=train_sequences.shape[-1],
            feature_dim=train_features.shape[-1],
            num_classes=train_cache.num_classes,
        ).to(device)
        ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None

        if args.class_weighting == "inverse":
            class_weight_values = compute_class_weights(train_labels)
        elif args.class_weighting == "effective":
            class_weight_values = compute_effective_number_weights(train_labels, beta=args.effective_beta)
        else:
            class_weight_values = None

        class_weights = None
        if class_weight_values is not None:
            class_weights = torch.as_tensor(class_weight_values, dtype=torch.float32, device=device)

        if args.loss == "focal":
            criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_f1 = -1.0
        best_predictions: np.ndarray | None = None
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_true, train_pred = run_epoch(model, train_loader, criterion, optimizer, device, ema=ema)
            eval_model = ema.module if ema is not None else model
            val_loss, val_true, val_pred = run_epoch(eval_model, val_loader, criterion, None, device)
            scheduler.step()

            train_f1 = macro_f1_score(train_true, train_pred, train_cache.num_classes)
            val_f1 = macro_f1_score(val_true, val_pred, train_cache.num_classes)
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_predictions = val_pred.copy()
                epochs_without_improvement = 0
                config = {
                    "sequence_dim": int(train_sequences.shape[-1]),
                    "feature_dim": int(train_features.shape[-1]),
                    "num_classes": int(train_cache.num_classes),
                    "fold_id": fold_id,
                    "seed": args.seed,
                    "loss": args.loss,
                    "cache_version": CACHE_VERSION,
                }
                save_checkpoint(output_dir, fold_id, eval_model, stats, config, val_f1)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_predictions is None:
            raise RuntimeError("Training did not produce validation predictions.")

        oof_predictions[val_idx] = best_predictions
        history.append(
            {
                "fold": fold_id,
                "train_size": int(train_idx.shape[0]),
                "val_size": int(val_idx.shape[0]),
                "best_val_f1": best_val_f1,
            }
        )
        print(f"Best val macro F1: {best_val_f1:.4f}")

    completed_mask = oof_predictions >= 0
    overall_f1 = macro_f1_score(
        train_cache.labels[completed_mask],
        oof_predictions[completed_mask],
        train_cache.num_classes,
    )
    print(f"\nOOF macro F1: {overall_f1:.4f}")

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "oof_macro_f1": overall_f1,
                "trained_folds": len(history),
                "args": vars(args),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
