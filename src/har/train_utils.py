from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
        false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
        false_negative = np.sum((y_true == class_id) & (y_pred != class_id))

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def make_stratified_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    label_values = np.unique(labels)
    num_classes = int(label_values.max()) + 1

    group_label_counts: dict[int, np.ndarray] = {}
    for group in unique_groups:
        group_labels = labels[groups == group]
        group_label_counts[int(group)] = np.bincount(group_labels, minlength=num_classes).astype(np.float32)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    sorted_groups = sorted(
        unique_groups,
        key=lambda group: (group_label_counts[int(group)].sum(), group_label_counts[int(group)].std()),
        reverse=True,
    )

    total_label_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    fold_label_counts = np.zeros((n_splits, num_classes), dtype=np.float32)
    fold_groups: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=np.float32)

    def evaluate_fold(fold_index: int, group_count: np.ndarray) -> float:
        fold_label_counts[fold_index] += group_count
        fold_sizes[fold_index] += group_count.sum()
        ratios = fold_label_counts / np.maximum(total_label_counts, 1.0)
        label_balance = ratios.std(axis=0).mean()
        size_balance = (fold_sizes / max(fold_sizes.sum(), 1.0)).std()
        score = float(label_balance + size_balance)
        fold_label_counts[fold_index] -= group_count
        fold_sizes[fold_index] -= group_count.sum()
        return score

    for group in sorted_groups:
        group_count = group_label_counts[int(group)]
        scores = [evaluate_fold(fold_index, group_count) for fold_index in range(n_splits)]
        best_fold = int(np.argmin(scores))
        fold_label_counts[best_fold] += group_count
        fold_sizes[best_fold] += group_count.sum()
        fold_groups[best_fold].append(int(group))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_splits):
        val_group_set = set(fold_groups[fold_index])
        val_mask = np.isin(groups, list(val_group_set))
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds.append((train_idx, val_idx))
    return folds


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist())
    total = float(labels.shape[0])
    num_classes = len(counts)
    return np.asarray([total / (num_classes * max(counts[class_id], 1)) for class_id in range(num_classes)], dtype=np.float32)


def compute_effective_number_weights(labels: np.ndarray, beta: float = 0.999) -> np.ndarray:
    counts = Counter(labels.tolist())
    num_classes = len(counts)
    raw_weights = []
    for class_id in range(num_classes):
        count = max(counts[class_id], 1)
        effective_num = 1.0 - beta**count
        raw_weights.append((1.0 - beta) / max(effective_num, 1e-12))

    weights = np.asarray(raw_weights, dtype=np.float32)
    weights = weights / max(weights.mean(), 1e-8)
    return weights


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        gathered_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gathered_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ce = -gathered_log_probs
        if self.weight is not None:
            ce = ce * self.weight[targets]
        focal_scale = torch.pow(1.0 - gathered_probs, self.gamma)
        return (focal_scale * ce).mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def summarize_label_distribution(labels: np.ndarray) -> dict[int, int]:
    distribution = defaultdict(int)
    for label, count in Counter(labels.tolist()).items():
        distribution[int(label)] = int(count)
    return dict(sorted(distribution.items()))
