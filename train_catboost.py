import argparse
from pathlib import Path

import numpy as np

from src.har.data import build_or_load_cache
from src.har.features import FeatureConfig
from src.har.train_utils import compute_class_weights, make_stratified_group_folds, macro_f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost on handcrafted features and save test probabilities."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the data directory (e.g., /content/drive/MyDrive/.../data)",
    )
    parser.add_argument(
        "--cache-dir",
        default="artifacts/cache",
        help="Directory used to store cached handcrafted features.",
    )
    parser.add_argument(
        "--output-probs",
        default="artifacts/catboost_test_probs.npy",
        help="Absolute or relative path to save the CatBoost test probabilities (.npy)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for CatBoost training"
    )
    parser.add_argument(
        "--iterations", type=int, default=500, help="Number of boosting iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Learning rate"
    )
    parser.add_argument(
        "--depth", type=int, default=6, help="Tree depth"
    )
    parser.add_argument(
        "--l2-leaf-reg", type=float, default=3.0, help="L2 regularization for leaves"
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Train only the first N folds for quick ablation smoke tests.",
    )
    parser.add_argument(
        "--skip-test-preds",
        action="store_true",
        help="Skip test-set probability generation and only report validation metrics.",
    )
    parser.add_argument(
        "--class-balance",
        choices=["balanced", "none", "manual"],
        default="balanced",
        help="Use balanced weights, disable class balancing, or provide manual weights.",
    )
    parser.add_argument(
        "--class-weight-boost",
        default="",
        help="Comma-separated label:multiplier boosts applied on top of balanced weights, e.g. '2:1.2,4:1.6,5:1.2'.",
    )
    parser.add_argument(
        "--manual-class-weights",
        default="",
        help="Comma-separated absolute class weights for labels 0..K-1, e.g. '1.0,1.0,2.0,1.0,3.0,2.0'. Used when --class-balance manual.",
    )
    parser.add_argument(
        "--disable-magnitude",
        action="store_true",
        help="Drop mean_mag and std_mag derived channels from the handcrafted feature pipeline.",
    )
    parser.add_argument(
        "--disable-base-stats",
        action="store_true",
        help="Drop per-channel global summary statistics.",
    )
    parser.add_argument(
        "--disable-diff-stats",
        action="store_true",
        help="Drop first-order difference statistics.",
    )
    parser.add_argument(
        "--disable-spectral",
        action="store_true",
        help="Drop FFT-based spectral features.",
    )
    parser.add_argument(
        "--disable-autocorr",
        action="store_true",
        help="Drop autocorrelation features.",
    )
    parser.add_argument(
        "--disable-chunk",
        action="store_true",
        help="Drop chunk-level temporal summary features.",
    )
    parser.add_argument(
        "--disable-correlations",
        action="store_true",
        help="Drop cross-axis correlation features.",
    )
    parser.add_argument(
        "--enable-advanced-features",
        "--enable-advanced-interactions",
        dest="enable_advanced_interactions",
        action="store_true",
        help="Add advanced cross-axis interaction channels that are not part of the original baseline.",
    )
    return parser.parse_args()


def build_feature_config(args: argparse.Namespace) -> FeatureConfig:
    return FeatureConfig(
        include_magnitude=not args.disable_magnitude,
        include_base_stats=not args.disable_base_stats,
        include_diff_stats=not args.disable_diff_stats,
        include_spectral=not args.disable_spectral,
        include_autocorr=not args.disable_autocorr,
        include_chunk=not args.disable_chunk,
        include_correlations=not args.disable_correlations,
        include_advanced_interactions=args.enable_advanced_interactions,
    )


def parse_class_weight_boost(spec: str) -> dict[int, float]:
    boosts: dict[int, float] = {}
    if not spec.strip():
        return boosts
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid class-weight boost item '{item}'. Expected format like '2:1.2'."
            )
        label_text, multiplier_text = item.split(":", 1)
        label = int(label_text.strip())
        multiplier = float(multiplier_text.strip())
        if multiplier <= 0.0:
            raise ValueError(f"Boost multiplier for label {label} must be positive.")
        boosts[label] = multiplier
    return boosts


def parse_manual_class_weights(spec: str, num_classes: int) -> list[float]:
    if not spec.strip():
        raise ValueError(
            "--manual-class-weights is required when --class-balance manual."
        )
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if len(values) != num_classes:
        raise ValueError(
            f"Expected {num_classes} manual class weights, got {len(values)}."
        )
    if any(value <= 0.0 for value in values):
        raise ValueError("All manual class weights must be positive.")
    return values


def resolve_class_weights(
    train_labels: np.ndarray,
    num_classes: int,
    balance_mode: str,
    boost_spec: str,
    manual_spec: str,
) -> list[float] | None:
    if balance_mode == "none":
        return None
    if balance_mode == "manual":
        return parse_manual_class_weights(manual_spec, num_classes)

    weights = compute_class_weights(train_labels).astype(np.float64)
    boosts = parse_class_weight_boost(boost_spec)
    for label, multiplier in boosts.items():
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} is out of range for {num_classes} classes.")
        weights[label] *= multiplier
    return weights.tolist()


def main() -> None:
    args = parse_args()
    feature_config = build_feature_config(args)
    from catboost import CatBoostClassifier

    print("Loading cache...")
    print(f"Feature config: {feature_config.to_dict()}")
    train_cache = build_or_load_cache(
        data_dir=args.data_dir,
        split="train",
        cache_dir=args.cache_dir,
        feature_config=feature_config,
    )
    test_cache = None
    if not args.skip_test_preds:
        test_cache = build_or_load_cache(
            data_dir=args.data_dir,
            split="test",
            cache_dir=args.cache_dir,
            feature_config=feature_config,
        )

    X_train, y_train, groups = (
        train_cache.features,
        train_cache.labels,
        train_cache.user_ids,
    )
    X_test = None if test_cache is None else test_cache.features
    num_classes = train_cache.num_classes
    print(f"Handcrafted feature dimension: {X_train.shape[1]}")
    if X_train.shape[1] == 0:
        raise ValueError("No handcrafted features remain after applying the requested ablation flags.")

    cat_test_probs = None
    if X_test is not None:
        cat_test_probs = np.zeros((len(X_test), num_classes), dtype=np.float64)
    oof_preds = np.full(len(X_train), -1, dtype=int)

    folds = make_stratified_group_folds(y_train, groups, n_splits=5, seed=args.seed)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("No folds selected. Check --max-folds.")
    num_folds = len(folds)

    print(f"Training CatBoost with seed {args.seed}...")
    for fold_id, (tr_idx, val_idx) in enumerate(folds):
        class_weights = resolve_class_weights(
            train_labels=y_train[tr_idx],
            num_classes=num_classes,
            balance_mode=args.class_balance,
            boost_spec=args.class_weight_boost,
            manual_spec=args.manual_class_weights,
        )
        if class_weights is not None:
            print(f"Fold {fold_id} class_weights: {class_weights}")

        clf = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            iterations=args.iterations,
            learning_rate=args.lr,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.seed,
            class_weights=class_weights,
            allow_writing_files=False,
            verbose=False,
        )

        clf.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=(X_train[val_idx], y_train[val_idx]),
            early_stopping_rounds=30,
            use_best_model=True,
        )

        val_probs = clf.predict_proba(X_train[val_idx])
        val_preds = np.argmax(val_probs, axis=1)
        oof_preds[val_idx] = val_preds

        fold_f1 = macro_f1_score(y_train[val_idx], val_preds, num_classes)
        print(f"Fold {fold_id} completed. Validation macro F1-score: {fold_f1:.4f}")

        if X_test is not None and cat_test_probs is not None:
            cat_test_probs += clf.predict_proba(X_test) / num_folds

    completed_mask = oof_preds >= 0
    overall_preds = oof_preds.copy()
    overall_preds[~completed_mask] = 0
    overall_f1 = macro_f1_score(y_train, overall_preds, num_classes)
    print(f"\nOverall OOF macro F1-score: {overall_f1:.4f}")
    if completed_mask.sum() < len(oof_preds):
        partial_f1 = macro_f1_score(
            y_train[completed_mask],
            oof_preds[completed_mask],
            num_classes,
        )
        print(f"Partial OOF macro F1-score (trained folds only): {partial_f1:.4f}")

    if cat_test_probs is not None:
        output_path = Path(args.output_probs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, cat_test_probs)
        print(f"Successfully saved CatBoost test probabilities to {output_path}")


if __name__ == "__main__":
    main()
