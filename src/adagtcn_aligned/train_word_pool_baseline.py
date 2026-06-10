"""Mask-aware pooled sklearn baselines for ZuCo word-level JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitSubjects:
    protocol: str
    train: list[str]
    val: list[str]
    test: list[str]
    note: str = ""


def load_split(split_json: Path, protocol: str) -> SplitSubjects:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    for split in data["splits"]:
        if split["protocol"] == protocol:
            return SplitSubjects(
                protocol=split["protocol"],
                train=list(split["train"]),
                val=list(split["val"]),
                test=list(split["test"]),
                note=split.get("note", ""),
            )
    available = [split["protocol"] for split in data["splits"]]
    raise ValueError("Protocol %s not found. Available: %s" % (protocol, available))


def valid_mask(record: dict[str, Any], modality: str, n_rows: int) -> np.ndarray:
    masks = record.get("masks") or {}
    missing = masks.get("%s_missing" % modality)
    if missing is None:
        return np.ones(n_rows, dtype=bool)
    mask = np.asarray(missing[:n_rows], dtype=np.int64) == 0
    if len(mask) < n_rows:
        mask = np.pad(mask, (0, n_rows - len(mask)), constant_values=False)
    return mask


def pool_modality(record: dict[str, Any], modality: str) -> np.ndarray:
    rows = record.get(modality) or []
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.zeros((0, 0), dtype=np.float32)

    n_rows = int(arr.shape[0])
    dim = int(arr.shape[1]) if arr.ndim == 2 and arr.shape else 0
    mask = valid_mask(record, modality, n_rows)
    if dim:
        mask = mask & np.isfinite(arr).all(axis=1)

    valid = arr[mask] if dim else np.zeros((0, 0), dtype=np.float32)
    valid_count = int(valid.shape[0])
    if valid_count:
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
    else:
        mean = np.zeros(dim, dtype=np.float32)
        std = np.zeros(dim, dtype=np.float32)

    valid_ratio = float(valid_count) / float(max(n_rows, 1))
    extras = np.asarray([valid_ratio, float(n_rows), float(valid_count)], dtype=np.float32)
    return np.concatenate([mean.astype(np.float32), std.astype(np.float32), extras])


def make_features(record: dict[str, Any], modality: str) -> np.ndarray:
    if modality == "concat":
        return np.concatenate([pool_modality(record, "eeg"), pool_modality(record, "gaze")])
    return pool_modality(record, modality)


def record_label(record: dict[str, Any]) -> int:
    if "y" in record:
        return int(record["y"])
    if "label" in record:
        return int(record["label"])
    raise KeyError("Record is missing both 'y' and 'label'.")


def load_examples(sequence_jsonl: Path, split: SplitSubjects, modality: str) -> tuple[dict[str, list[np.ndarray]], dict[str, list[int]]]:
    roles_by_subject = {subject: "train" for subject in split.train}
    roles_by_subject.update({subject: "val" for subject in split.val})
    roles_by_subject.update({subject: "test" for subject in split.test})

    features: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    expected_dim: int | None = None

    with sequence_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            role = roles_by_subject.get(record.get("subject"))
            if role is None:
                continue

            vector = make_features(record, modality)
            if expected_dim is None:
                expected_dim = int(vector.shape[0])
            elif int(vector.shape[0]) != expected_dim:
                raise ValueError(
                    "Inconsistent feature dimension for %s: got %d, expected %d"
                    % (record.get("sequence_id", "<unknown>"), vector.shape[0], expected_dim)
                )

            features[role].append(vector)
            labels[role].append(record_label(record))

    return features, labels


def stack_examples(features: list[np.ndarray], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


def evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(x)
    classes = model.named_steps["classifier"].classes_
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
        prob = model.predict_proba(x)[:, pos_idx]
    else:
        prob = np.zeros(len(y), dtype=np.float32)

    if len(set(y.tolist())) < 2:
        auroc = 0.5
    else:
        auroc = roc_auc_score(y, prob)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(auroc),
        "n": int(len(y)),
    }


def write_csv(row: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mask-aware pooled LR baselines on word-level ZuCo JSONL.")
    parser.add_argument("--sequence-jsonl", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=Path("reports/adagtcn_aligned/subject_splits.json"))
    parser.add_argument("--protocol", default="Y16_LOSO_YAC")
    parser.add_argument("--modality", choices=["eeg", "gaze", "concat"], default="gaze")
    parser.add_argument("--model", default="word_pool_lr")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adagtcn_word_pool_baseline"))
    args = parser.parse_args()

    split = load_split(args.split_json, args.protocol)
    features, labels = load_examples(args.sequence_jsonl, split, args.modality)
    x_train, y_train = stack_examples(features["train"], labels["train"])
    x_val, y_val = stack_examples(features["val"], labels["val"])
    x_test, y_test = stack_examples(features["test"], labels["test"])

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise RuntimeError(
            "Empty train/val/test split: train=%d val=%d test=%d"
            % (len(y_train), len(y_val), len(y_test))
        )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    val_metrics = evaluate(model, x_val, y_val)
    test_metrics = evaluate(model, x_test, y_test)

    row: dict[str, Any] = {
        "protocol": args.protocol,
        "model": args.model,
        "modality": args.modality,
        "train_examples": int(len(y_train)),
        "val_examples": int(len(y_val)),
        "test_examples": int(len(y_test)),
        "feature_dim": int(x_train.shape[1]),
    }
    row.update({"val_" + key: value for key, value in val_metrics.items()})
    row.update({"test_" + key: value for key, value in test_metrics.items()})

    stem = "%s_%s_%s" % (args.model, args.modality, args.protocol)
    csv_path = args.output_dir / ("%s.csv" % stem)
    json_path = args.output_dir / ("%s_meta.json" % stem)
    write_csv(row, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "result": row,
                "split": asdict(split),
                "sequence_jsonl": str(args.sequence_jsonl),
                "pooling": "valid-token mean + std with valid_ratio, sequence_length, valid_count",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(row, indent=2), flush=True)
    print("Wrote %s" % csv_path)
    print("Wrote %s" % json_path)


if __name__ == "__main__":
    main()
