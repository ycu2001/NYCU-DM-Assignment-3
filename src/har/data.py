from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    BASE_COLUMNS,
    DEFAULT_FEATURE_CONFIG,
    FeatureConfig,
    augment_sequence,
    extract_global_features,
)

CACHE_VERSION = "v3"


@dataclass(frozen=True)
class HARCache:
    sequences: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    file_ids: np.ndarray
    user_ids: np.ndarray
    sequence_columns: list[str]

    @property
    def num_classes(self) -> int:
        valid_labels = self.labels[self.labels >= 0]
        return int(valid_labels.max()) + 1 if valid_labels.size else 0


class HARDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = None if labels is None else torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        if self.labels is None:
            return self.sequences[index], self.features[index]
        return self.sequences[index], self.features[index], self.labels[index]


def _extract_user_id(file_path: Path) -> int:
    return int(file_path.parent.name.split("_")[1])


def _scan_csv_files(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.rglob("*.csv") if path.name.endswith(".csv"))


def _build_cache(split_dir: Path, feature_config: FeatureConfig) -> HARCache:
    files = _scan_csv_files(split_dir)
    sequences: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    file_ids: list[int] = []
    user_ids: list[int] = []

    for file_path in files:
        frame = pd.read_csv(file_path)
        raw_sequence = frame[BASE_COLUMNS].to_numpy(dtype=np.float32)
        sequences.append(augment_sequence(raw_sequence))
        features.append(extract_global_features(raw_sequence, config=feature_config))
        if "label" in frame.columns:
            labels.append(int(frame["label"].iloc[0]))
        else:
            labels.append(-1)
        if "file_id" in frame.columns:
            file_ids.append(int(frame["file_id"].iloc[0]))
        else:
            file_ids.append(int(file_path.stem))
        user_ids.append(_extract_user_id(file_path))

    sequence_columns = (
        BASE_COLUMNS
        + ["mean_mag", "std_mag"]
        + [f"diff_{column}" for column in BASE_COLUMNS]
        + ["diff_mean_mag", "diff_std_mag"]
    )
    return HARCache(
        sequences=np.stack(sequences).astype(np.float32),
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        file_ids=np.asarray(file_ids, dtype=np.int64),
        user_ids=np.asarray(user_ids, dtype=np.int64),
        sequence_columns=sequence_columns,
    )


def _cache_stem(split: str, feature_config: FeatureConfig) -> str:
    if feature_config == DEFAULT_FEATURE_CONFIG:
        return f"{split}_cache"
    return f"{split}_cache_{feature_config.cache_tag()}"


def build_or_load_cache(
    data_dir: str | Path,
    split: str,
    cache_dir: str | Path = "artifacts/cache",
    feature_config: FeatureConfig | None = None,
) -> HARCache:
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_config = DEFAULT_FEATURE_CONFIG if feature_config is None else feature_config

    if split == "train":
        split_dir = data_dir / "train" / "train"
    elif split == "test":
        split_dir = data_dir / "test" / "test"
    else:
        raise ValueError(f"Unsupported split: {split}")

    cache_stem = _cache_stem(split, feature_config)
    cache_path = cache_dir / f"{cache_stem}.npz"
    meta_path = cache_dir / f"{cache_stem}_meta.json"

    if cache_path.exists() and meta_path.exists():
        bundle = np.load(cache_path)
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if (
            meta.get("cache_version") == CACHE_VERSION
            and meta.get("feature_config") == feature_config.to_dict()
        ):
            return HARCache(
                sequences=bundle["sequences"],
                features=bundle["features"],
                labels=bundle["labels"],
                file_ids=bundle["file_ids"],
                user_ids=bundle["user_ids"],
                sequence_columns=meta["sequence_columns"],
            )

    cache = _build_cache(split_dir, feature_config)
    np.savez_compressed(
        cache_path,
        sequences=cache.sequences,
        features=cache.features,
        labels=cache.labels,
        file_ids=cache.file_ids,
        user_ids=cache.user_ids,
    )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_version": CACHE_VERSION,
                "sequence_columns": cache.sequence_columns,
                "feature_config": feature_config.to_dict(),
                "feature_dim": int(cache.features.shape[1]),
            },
            handle,
            indent=2,
        )
    return cache
