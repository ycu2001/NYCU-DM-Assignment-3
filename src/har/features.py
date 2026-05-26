from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


BASE_COLUMNS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
AUTOCORR_LAGS = (1, 5, 10, 30, 60)
CHUNK_SIZES = (30, 60)


@dataclass(frozen=True)
class FeatureConfig:
    include_magnitude: bool = True
    include_base_stats: bool = True
    include_diff_stats: bool = True
    include_spectral: bool = True
    include_autocorr: bool = True
    include_chunk: bool = True
    include_correlations: bool = True
    include_advanced_interactions: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def cache_tag(self) -> str:
        if self == DEFAULT_FEATURE_CONFIG:
            return "default"
        return "_".join(
            [
                f"mag{int(self.include_magnitude)}",
                f"base{int(self.include_base_stats)}",
                f"diff{int(self.include_diff_stats)}",
                f"spec{int(self.include_spectral)}",
                f"auto{int(self.include_autocorr)}",
                f"chunk{int(self.include_chunk)}",
                f"corr{int(self.include_correlations)}",
                f"adv{int(self.include_advanced_interactions)}",
            ]
        )


DEFAULT_FEATURE_CONFIG = FeatureConfig()


def _safe_std(values: np.ndarray) -> float:
    std = float(values.std())
    return std if std > 1e-8 else 0.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_std = _safe_std(a)
    b_std = _safe_std(b)
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _slope(values: np.ndarray) -> float:
    t = np.arange(values.shape[0], dtype=np.float32)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom == 0.0:
        return 0.0
    centered = values - values.mean()
    return float(np.dot(t, centered) / denom)


def _spectral_features(values: np.ndarray) -> list[float]:
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.shape[0] <= 1:
        return [0.0, 0.0, 0.0, 0.0]
    spectrum = spectrum[1:]
    total = float(spectrum.sum())
    if total <= 1e-8:
        return [0.0, 0.0, 0.0, 0.0]
    freqs = np.arange(1, spectrum.shape[0] + 1, dtype=np.float32)
    top_indices = np.argsort(spectrum)[-2:]
    top_values = spectrum[top_indices]
    top_values.sort()
    dominant_freq = float(freqs[int(np.argmax(spectrum))])
    spectral_centroid = float(np.dot(freqs, spectrum) / total)
    return [
        dominant_freq,
        spectral_centroid,
        float(top_values[-1] / total),
        float(top_values[-2] / total) if top_values.shape[0] > 1 else 0.0,
    ]


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= values.shape[0]:
        return 0.0
    left = values[:-lag] - values[:-lag].mean()
    right = values[lag:] - values[lag:].mean()
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def _chunk_features(values: np.ndarray, chunk_size: int) -> list[float]:
    usable_length = (values.shape[0] // chunk_size) * chunk_size
    if usable_length == 0:
        num_chunks = max(values.shape[0] // max(chunk_size, 1), 1)
        return [0.0] * (2 * num_chunks + 11)

    chunks = values[:usable_length].reshape(-1, chunk_size)
    chunk_means = chunks.mean(axis=1)
    chunk_stds = chunks.std(axis=1)
    chunk_ranges = chunks.max(axis=1) - chunks.min(axis=1)

    features = chunk_means.tolist()
    features.extend(chunk_stds.tolist())
    features.extend(
        [
            float(chunk_means.mean()),
            float(chunk_means.std()),
            float(chunk_means.min()),
            float(chunk_means.max()),
            float(chunk_stds.mean()),
            float(chunk_stds.std()),
            float(chunk_stds.min()),
            float(chunk_stds.max()),
            float(chunk_ranges.mean()),
            float(chunk_ranges.std()),
            float(chunk_ranges.max()),
        ]
    )
    return features


def _summarize_channel(channel: np.ndarray, config: FeatureConfig) -> list[float]:
    features: list[float] = []
    diffs = np.diff(channel)
    if diffs.size == 0:
        diffs = np.zeros(1, dtype=np.float32)

    if config.include_base_stats:
        q10, q25, q75, q90 = np.percentile(channel, [10, 25, 75, 90])
        features.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(channel.min()),
                float(channel.max()),
                float(np.median(channel)),
                float(q10),
                float(q25),
                float(q75),
                float(q90),
                float(channel.max() - channel.min()),
                float(np.sqrt(np.mean(np.square(channel)))),
                float(np.mean(np.abs(channel))),
                float(channel[0]),
                float(channel[-1]),
                float(channel[-1] - channel[0]),
                _slope(channel),
                float(np.mean(np.square(channel))),
            ]
        )

    if config.include_diff_stats:
        features.extend(
            [
                float(diffs.mean()),
                float(diffs.std()),
                float(np.mean(np.abs(diffs))),
                float(np.max(np.abs(diffs))),
                float(np.mean(np.square(diffs))),
            ]
        )

    if config.include_spectral:
        features.extend(_spectral_features(channel))

    if config.include_autocorr:
        for lag in AUTOCORR_LAGS:
            features.append(_autocorrelation(channel, lag))

    if config.include_chunk:
        for chunk_size in CHUNK_SIZES:
            features.extend(_chunk_features(channel, chunk_size))

    return features


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    mean_xyz = sequence[:, :3]
    std_xyz = sequence[:, 3:6]
    mean_mag = np.linalg.norm(mean_xyz, axis=1, keepdims=True)
    std_mag = np.linalg.norm(std_xyz, axis=1, keepdims=True)
    first_diff = np.diff(sequence, axis=0, prepend=sequence[:1])
    mag_diff = np.diff(np.concatenate([mean_mag, std_mag], axis=1), axis=0, prepend=np.zeros((1, 2), dtype=sequence.dtype))
    return np.concatenate([sequence, mean_mag, std_mag, first_diff, mag_diff], axis=1).astype(np.float32)


def extract_global_features(
    sequence: np.ndarray,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> np.ndarray:
    mean_xyz = sequence[:, :3]
    std_xyz = sequence[:, 3:6]
    mean_mag = np.linalg.norm(mean_xyz, axis=1, keepdims=True)
    std_mag = np.linalg.norm(std_xyz, axis=1, keepdims=True)

    channel_map: dict[str, np.ndarray] = {
        "mean_x": sequence[:, 0],
        "mean_y": sequence[:, 1],
        "mean_z": sequence[:, 2],
        "std_x": sequence[:, 3],
        "std_y": sequence[:, 4],
        "std_z": sequence[:, 5],
    }
    if config.include_magnitude:
        channel_map["mean_mag"] = mean_mag[:, 0]
        channel_map["std_mag"] = std_mag[:, 0]

    features: list[float] = []
    for channel in channel_map.values():
        features.extend(_summarize_channel(channel, config))

    if config.include_advanced_interactions:
        advanced_channels = [
            mean_xyz[:, 0] * mean_xyz[:, 1],
            mean_xyz[:, 0] * mean_xyz[:, 2],
            mean_xyz[:, 1] * mean_xyz[:, 2],
            std_xyz[:, 0] * std_xyz[:, 1],
            std_xyz[:, 0] * std_xyz[:, 2],
            std_xyz[:, 1] * std_xyz[:, 2],
        ]
        for channel in advanced_channels:
            features.extend(_summarize_channel(channel, config))

    if config.include_correlations:
        correlation_pairs = [
            ("mean_x", "mean_y"),
            ("mean_x", "mean_z"),
            ("mean_y", "mean_z"),
            ("std_x", "std_y"),
            ("std_x", "std_z"),
            ("std_y", "std_z"),
            ("mean_x", "std_x"),
            ("mean_y", "std_y"),
            ("mean_z", "std_z"),
        ]
        if config.include_magnitude:
            correlation_pairs.append(("mean_mag", "std_mag"))

        for left, right in correlation_pairs:
            features.append(_safe_corr(channel_map[left], channel_map[right]))

    if not features:
        raise ValueError("FeatureConfig disabled every handcrafted feature group.")

    return np.asarray(features, dtype=np.float32)
