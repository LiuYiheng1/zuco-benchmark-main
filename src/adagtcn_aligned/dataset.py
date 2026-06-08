"""Dataset utilities for AdaGTCN-aligned word/fixation experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


GAZE_SCALAR_FIELDS = ["FFD", "GD", "GPT", "TRT", "nFixations", "meanPupilSize"]


@dataclass
class SplitSubjects:
    protocol: str
    train: list[str]
    val: list[str]
    test: list[str]
    note: str = ""


@dataclass
class FeatureStats:
    eeg_mean: np.ndarray
    eeg_std: np.ndarray
    gaze_mean: np.ndarray
    gaze_std: np.ndarray


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
    available = [s["protocol"] for s in data["splits"]]
    raise ValueError("Protocol %s not found. Available: %s" % (protocol, available))


def read_jsonl_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def read_record_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(offset)
        line = f.readline()
    return json.loads(line.decode("utf-8"))


def gaze_to_vector(gaze: dict[str, Any], position: int, length: int) -> np.ndarray:
    values = []
    for field in GAZE_SCALAR_FIELDS:
        value = gaze.get(field)
        if value is None:
            value = 0.0
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        if field in {"FFD", "GD", "GPT", "TRT", "meanPupilSize"}:
            value = math.log1p(max(value, 0.0))
        values.append(value)

    fix_positions = gaze.get("fixPositions") or []
    valid_positions = []
    for x in fix_positions:
        value = float(x)
        if math.isfinite(value) and value > 0:
            valid_positions.append(value)
    first_pos = valid_positions[0] if valid_positions else 0.0
    last_pos = valid_positions[-1] if valid_positions else 0.0
    n_positions = float(len(valid_positions))
    regression = 0.0
    if len(valid_positions) >= 2:
        regression = float(sum(1 for a, b in zip(valid_positions, valid_positions[1:]) if b < a))

    rel_pos = float(position) / max(float(length - 1), 1.0)
    values.extend([first_pos, last_pos, n_positions, regression, rel_pos])
    return np.asarray(values, dtype=np.float32)


def infer_dims(jsonl_path: Path, offsets: Iterable[int]) -> tuple[int, int]:
    eeg_dim = 0
    gaze_dim = len(GAZE_SCALAR_FIELDS) + 5
    for offset in offsets:
        record = read_record_at(jsonl_path, offset)
        for row in record.get("eeg", []):
            if row:
                eeg_dim = max(eeg_dim, len(row))
        if eeg_dim:
            break
    return eeg_dim, gaze_dim


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0 or self.dim == 0:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        mask = np.isfinite(x).all(axis=1)
        x = x[mask]
        if len(x) == 0:
            return
        self.count += len(x)
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.dim == 0:
            return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
        if self.count == 0:
            return np.zeros(self.dim, dtype=np.float32), np.ones(self.dim, dtype=np.float32)
        mean = self.sum / float(self.count)
        var = self.sumsq / float(self.count) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-6))
        return mean.astype(np.float32), std.astype(np.float32)


def collect_offsets(jsonl_path: Path, subjects: Iterable[str]) -> list[dict[str, Any]]:
    subject_set = set(subjects)
    rows = []
    for offset in read_jsonl_offsets(jsonl_path):
        record = read_record_at(jsonl_path, offset)
        if record["subject"] in subject_set:
            rows.append(
                {
                    "offset": offset,
                    "sequence_id": record["sequence_id"],
                    "subject": record["subject"],
                    "y": int(record["y"]),
                }
            )
    return rows


def fit_feature_stats(jsonl_path: Path, examples: list[dict[str, Any]], eeg_dim: int, gaze_dim: int) -> FeatureStats:
    eeg_stats = RunningMoments(eeg_dim)
    gaze_stats = RunningMoments(gaze_dim)

    for example in examples:
        record = read_record_at(jsonl_path, int(example["offset"]))
        length = len(record.get("words", []))
        eeg_rows = record.get("eeg", [])
        eeg_masks = record.get("eeg_mask", [])
        gaze_rows = record.get("gaze", [])
        gaze_masks = record.get("gaze_mask", [])
        for idx in range(length):
            if idx < len(gaze_rows) and idx < len(gaze_masks) and gaze_masks[idx]:
                gaze_stats.update(gaze_to_vector(gaze_rows[idx], idx, length)[None, :])
            if eeg_dim and idx < len(eeg_rows) and idx < len(eeg_masks) and eeg_masks[idx] and eeg_rows[idx]:
                row = np.zeros(eeg_dim, dtype=np.float32)
                values = np.asarray(eeg_rows[idx], dtype=np.float32)
                n = min(eeg_dim, len(values))
                row[:n] = values[:n]
                eeg_stats.update(row[None, :])

    eeg_mean, eeg_std = eeg_stats.finalize()
    gaze_mean, gaze_std = gaze_stats.finalize()
    return FeatureStats(eeg_mean=eeg_mean, eeg_std=eeg_std, gaze_mean=gaze_mean, gaze_std=gaze_std)


class ZuCoSequenceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        examples: list[dict[str, Any]],
        subject_to_idx: dict[str, int],
        eeg_dim: int,
        gaze_dim: int,
        stats: FeatureStats,
        max_len: int = 80,
        cache_records: bool = False,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.examples = examples
        self.subject_to_idx = subject_to_idx
        self.eeg_dim = eeg_dim
        self.gaze_dim = gaze_dim
        self.stats = stats
        self.max_len = max_len
        self.records = None
        if cache_records:
            self.records = [read_record_at(self.jsonl_path, int(example["offset"])) for example in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if self.records is None:
            record = read_record_at(self.jsonl_path, int(example["offset"]))
        else:
            record = self.records[index]
        n_words = len(record.get("words", []))
        length = min(n_words, self.max_len)

        eeg = np.zeros((self.max_len, self.eeg_dim), dtype=np.float32)
        gaze = np.zeros((self.max_len, self.gaze_dim), dtype=np.float32)
        eeg_mask = np.zeros(self.max_len, dtype=np.float32)
        gaze_mask = np.zeros(self.max_len, dtype=np.float32)
        step_mask = np.zeros(self.max_len, dtype=np.float32)

        eeg_rows = record.get("eeg", [])
        eeg_masks = record.get("eeg_mask", [])
        gaze_rows = record.get("gaze", [])
        gaze_masks = record.get("gaze_mask", [])

        for idx in range(length):
            step_mask[idx] = 1.0
            if idx < len(gaze_rows) and idx < len(gaze_masks) and gaze_masks[idx]:
                gv = gaze_to_vector(gaze_rows[idx], idx, n_words)
                gv = (gv - self.stats.gaze_mean) / self.stats.gaze_std
                gaze[idx] = gv
                gaze_mask[idx] = 1.0

            if self.eeg_dim and idx < len(eeg_rows) and idx < len(eeg_masks) and eeg_masks[idx] and eeg_rows[idx]:
                ev = np.zeros(self.eeg_dim, dtype=np.float32)
                values = np.asarray(eeg_rows[idx], dtype=np.float32)
                n = min(self.eeg_dim, len(values))
                ev[:n] = values[:n]
                ev = (ev - self.stats.eeg_mean) / self.stats.eeg_std
                eeg[idx] = ev
                eeg_mask[idx] = 1.0

        return {
            "eeg": torch.from_numpy(eeg),
            "gaze": torch.from_numpy(gaze),
            "eeg_mask": torch.from_numpy(eeg_mask),
            "gaze_mask": torch.from_numpy(gaze_mask),
            "step_mask": torch.from_numpy(step_mask),
            "y": torch.tensor(int(record["y"]), dtype=torch.long),
            "subject": torch.tensor(self.subject_to_idx[record["subject"]], dtype=torch.long),
            "sequence_id": record["sequence_id"],
        }


def make_datasets(
    jsonl_path: Path,
    split: SplitSubjects,
    max_len: int,
    debug_random_split: bool = False,
    cache_records: bool = False,
) -> tuple[ZuCoSequenceDataset, Optional[ZuCoSequenceDataset], ZuCoSequenceDataset, dict[str, Any]]:
    all_subjects = sorted(set(split.train + split.val + split.test))
    subject_to_idx = {subject: idx for idx, subject in enumerate(all_subjects)}

    if debug_random_split:
        examples = collect_offsets(jsonl_path, all_subjects)
        rng = np.random.RandomState(0)
        order = rng.permutation(len(examples))
        n_train = max(1, int(0.6 * len(examples)))
        n_val = max(1, int(0.2 * len(examples))) if len(examples) >= 5 else 0
        train_examples = [examples[i] for i in order[:n_train]]
        val_examples = [examples[i] for i in order[n_train:n_train + n_val]]
        test_examples = [examples[i] for i in order[n_train + n_val:]]
        if not test_examples:
            test_examples = val_examples or train_examples
    else:
        train_examples = collect_offsets(jsonl_path, split.train)
        val_examples = collect_offsets(jsonl_path, split.val)
        test_examples = collect_offsets(jsonl_path, split.test)

    all_offsets = [ex["offset"] for ex in train_examples + val_examples + test_examples]
    eeg_dim, gaze_dim = infer_dims(jsonl_path, all_offsets)
    stats = fit_feature_stats(jsonl_path, train_examples, eeg_dim, gaze_dim)

    train = ZuCoSequenceDataset(
        jsonl_path,
        train_examples,
        subject_to_idx,
        eeg_dim,
        gaze_dim,
        stats,
        max_len=max_len,
        cache_records=cache_records,
    )
    val = None
    if val_examples:
        val = ZuCoSequenceDataset(
            jsonl_path,
            val_examples,
            subject_to_idx,
            eeg_dim,
            gaze_dim,
            stats,
            max_len=max_len,
            cache_records=cache_records,
        )
    test = ZuCoSequenceDataset(
        jsonl_path,
        test_examples,
        subject_to_idx,
        eeg_dim,
        gaze_dim,
        stats,
        max_len=max_len,
        cache_records=cache_records,
    )

    meta = {
        "subject_to_idx": subject_to_idx,
        "eeg_dim": eeg_dim,
        "gaze_dim": gaze_dim,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "n_test": len(test_examples),
        "debug_random_split": debug_random_split,
        "cache_records": cache_records,
    }
    return train, val, test, meta
