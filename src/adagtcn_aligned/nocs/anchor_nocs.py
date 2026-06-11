"""Exact sklearn-style gaze anchors for NOCS safe admission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.adagtcn_aligned.nocs.dataset_nocs import SplitSubjects, record_label


@dataclass
class AnchorExamples:
    features: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    rows: dict[str, list[dict[str, Any]]]
    raw_token_dim: int
    feature_dim: int
    component_raw_token_dims: dict[str, int]


def token_array(record: dict[str, Any], modality: str) -> np.ndarray:
    arr = np.asarray(record.get(modality) or [], dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    return arr


def modality_valid_mask(record: dict[str, Any], modality: str, n_rows: int) -> np.ndarray:
    masks = record.get("masks") or {}
    missing = masks.get("%s_missing" % modality)
    if missing is None:
        mask = np.ones(n_rows, dtype=bool)
    else:
        mask = np.asarray(missing[:n_rows], dtype=np.int64) == 0
        if len(mask) < n_rows:
            mask = np.pad(mask, (0, n_rows - len(mask)), constant_values=False)
    return mask


def gaze_features(record: dict[str, Any]) -> tuple[np.ndarray, int, dict[str, int]]:
    gaze = token_array(record, "gaze")
    eeg = token_array(record, "eeg")
    n_rows = int(gaze.shape[0])
    raw_dim = int(gaze.shape[1]) if gaze.ndim == 2 and gaze.shape else 0
    if raw_dim:
        mask = modality_valid_mask(record, "gaze", n_rows) & np.isfinite(gaze).all(axis=1)
        valid = gaze[mask]
    else:
        mask = np.zeros(n_rows, dtype=bool)
        valid = np.zeros((0, 0), dtype=np.float32)

    valid_count = int(valid.shape[0])
    if valid_count:
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
    else:
        mean = np.zeros(raw_dim, dtype=np.float32)
        std = np.zeros(raw_dim, dtype=np.float32)
    valid_ratio = float(valid_count) / float(max(n_rows, 1))
    extras = np.asarray([valid_ratio, float(n_rows), float(valid_count)], dtype=np.float32)
    feature = np.concatenate([mean.astype(np.float32), std.astype(np.float32), extras])
    return feature, raw_dim, {"eeg": int(eeg.shape[1]) if eeg.ndim == 2 and eeg.shape else 0, "gaze": raw_dim}


def label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def load_anchor_examples(sequence_jsonl: Path, split: SplitSubjects) -> AnchorExamples:
    roles_by_subject = {subject: "train" for subject in split.train}
    roles_by_subject.update({subject: "val" for subject in split.val})
    roles_by_subject.update({subject: "test" for subject in split.test})

    feature_lists: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    label_lists: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    expected_dim: int | None = None
    expected_raw_dim: int | None = None
    component_dims: dict[str, int] = {"eeg": 0, "gaze": 0}

    with sequence_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            subject = str(record.get("subject", ""))
            role = roles_by_subject.get(subject)
            if role is None:
                continue
            feature, raw_dim, dims = gaze_features(record)
            if expected_dim is None:
                expected_dim = int(feature.shape[0])
                expected_raw_dim = raw_dim
                component_dims = dims
            elif int(feature.shape[0]) != expected_dim:
                raise RuntimeError(
                    "Inconsistent gaze feature dimension for %s: got %d, expected %d"
                    % (record.get("sequence_id", "<unknown>"), feature.shape[0], expected_dim)
                )
            elif raw_dim != expected_raw_dim:
                raise RuntimeError(
                    "Inconsistent gaze raw token dimension for %s: got %d, expected %d"
                    % (record.get("sequence_id", "<unknown>"), raw_dim, expected_raw_dim)
                )
            label = record_label(record)
            feature_lists[role].append(feature)
            label_lists[role].append(label)
            rows[role].append(
                {
                    "sequence_id": record.get("sequence_id", ""),
                    "subject": subject,
                    "y_true": int(label),
                }
            )

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for role in ["train", "val", "test"]:
        if feature_lists[role]:
            features[role] = np.vstack(feature_lists[role]).astype(np.float32)
        else:
            features[role] = np.zeros((0, int(expected_dim or 0)), dtype=np.float32)
        labels[role] = np.asarray(label_lists[role], dtype=np.int64)
    if any(labels[role].size == 0 for role in ["train", "val", "test"]):
        raise RuntimeError(
            "Empty train/val/test split: train=%d val=%d test=%d"
            % (labels["train"].size, labels["val"].size, labels["test"].size)
        )
    feature_dim = int(expected_dim or 0)
    raw_token_dim = int(expected_raw_dim or 0)
    expected_feature_dim = 2 * raw_token_dim + 3
    if feature_dim != expected_feature_dim:
        raise RuntimeError(
            "Pooled gaze feature dimension mismatch: feature_dim=%d raw_token_dim=%d expected=%d"
            % (feature_dim, raw_token_dim, expected_feature_dim)
        )
    return AnchorExamples(
        features=features,
        labels=labels,
        rows=rows,
        raw_token_dim=raw_token_dim,
        feature_dim=feature_dim,
        component_raw_token_dims=component_dims,
    )


def make_anchor_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            ),
        ]
    )


def positive_prob(model: Pipeline, x: np.ndarray) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_
    probs = model.predict_proba(x)
    if 1 not in classes:
        return np.zeros(x.shape[0], dtype=np.float32)
    pos_idx = int(np.where(classes == 1)[0][0])
    return probs[:, pos_idx].astype(np.float64)


def prob_to_pred(prob: np.ndarray) -> np.ndarray:
    return (prob >= 0.5).astype(np.int64)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = prob_to_pred(y_prob)
    if len(set(y_true.astype(int).tolist())) < 2:
        auroc = 0.5
        auprc = float(np.mean(y_true)) if len(y_true) else 0.0
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "auprc": auprc,
        "n": int(len(y_true)),
    }


def prediction_rows(
    base_rows: list[dict[str, Any]],
    split_name: str,
    protocol: str,
    heldout_subject: str,
    prob: np.ndarray,
    prefix: str,
) -> list[dict[str, Any]]:
    pred = prob_to_pred(prob)
    out = []
    for idx, row in enumerate(base_rows):
        item = dict(row)
        item["split"] = split_name
        item["protocol"] = protocol
        item["heldout_subject"] = heldout_subject
        item["y_prob_%s" % prefix] = float(prob[idx])
        item["y_pred_%s" % prefix] = int(pred[idx])
        out.append(item)
    return out


def safe_mix(anchor_prob: np.ndarray, corrected_prob: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * anchor_prob + alpha * corrected_prob


def select_safe_alpha(
    y_val: np.ndarray,
    anchor_val_prob: np.ndarray,
    corrected_val_prob: np.ndarray,
    alphas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> tuple[float, str, dict[str, float]]:
    anchor_val = binary_metrics(y_val, anchor_val_prob)["auroc"]
    corrected_val = binary_metrics(y_val, corrected_val_prob)["auroc"]
    if corrected_val <= anchor_val:
        return 0.0, "anchor_only", {
            "anchor_val_auroc": anchor_val,
            "corrected_val_auroc": corrected_val,
            "safe_val_auroc": anchor_val,
        }
    best_alpha = alphas[0]
    best_auroc = -1.0
    for alpha in alphas:
        value = binary_metrics(y_val, safe_mix(anchor_val_prob, corrected_val_prob, alpha))["auroc"]
        if value > best_auroc:
            best_alpha = alpha
            best_auroc = value
    return best_alpha, "mixture", {
        "anchor_val_auroc": anchor_val,
        "corrected_val_auroc": corrected_val,
        "safe_val_auroc": best_auroc,
    }


def sklearn_model_metadata(model: Pipeline) -> dict[str, Any]:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "lr_coef": classifier.coef_.astype(float).tolist(),
        "lr_intercept": classifier.intercept_.astype(float).tolist(),
        "class_weight": classifier.class_weight,
        "solver": classifier.solver,
        "max_iter": int(classifier.max_iter),
        "classes": classifier.classes_.astype(int).tolist(),
    }
