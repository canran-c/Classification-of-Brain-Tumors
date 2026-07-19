from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    accuracy = float(np.mean(y_true == y_pred))
    precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    num_classes = y_prob.shape[1]
    auc_scores = []
    for cls in range(num_classes):
        y_bin = (y_true == cls).astype(np.int32)
        scores = y_prob[:, cls]
        auc = _binary_auc(y_bin, scores)
        if not np.isnan(auc):
            auc_scores.append(auc)
    roc_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_ovr_macro": roc_auc,
    }


def _binary_auc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    pos_count = int(np.sum(y_true_binary == 1))
    neg_count = int(np.sum(y_true_binary == 0))
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    ranks = np.argsort(np.argsort(y_score)) + 1
    pos_ranks_sum = float(np.sum(ranks[y_true_binary == 1]))
    auc = (pos_ranks_sum - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)
    return float(auc)


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
