"""Dataset utilities for NOCS word-level EEG+gaze models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
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


def record_label(record: dict[str, Any]) -> int:
    if "y" in record:
        return int(record["y"])
    if "label" in record:
        return int(record["label"])
    raise KeyError("Record is missing both 'y' and 'label'.")


def collect_examples(jsonl_path: Path, subjects: list[str], role: str) -> list[dict[str, Any]]:
    subject_set = set(subjects)
    examples = []
    for offset in read_jsonl_offsets(jsonl_path):
        record = read_record_at(jsonl_path, offset)
        subject = record.get("subject")
        if subject not in subject_set:
            continue
        examples.append(
            {
                "offset": offset,
                "role": role,
                "subject": subject,
                "sequence_id": record.get("sequence_id", ""),
                "label": record_label(record),
            }
        )
    return examples


def infer_dims(jsonl_path: Path, examples: list[dict[str, Any]]) -> tuple[int, int]:
    eeg_dim, gaze_dim = 0, 0
    for example in examples:
        record = read_record_at(jsonl_path, int(example["offset"]))
        eeg = record.get("eeg") or []
        gaze = record.get("gaze") or []
        if eeg and not eeg_dim:
            eeg_dim = len(eeg[0])
        if gaze and not gaze_dim:
            gaze_dim = len(gaze[0])
        if eeg_dim and gaze_dim:
            break
    if not eeg_dim or not gaze_dim:
        raise RuntimeError("Could not infer eeg/gaze dimensions from %s" % jsonl_path)
    return eeg_dim, gaze_dim


class NOCSSequenceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        examples: list[dict[str, Any]],
        subject_to_idx: dict[str, int],
        eeg_dim: int,
        gaze_dim: int,
        cache_records: bool = False,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.examples = examples
        self.subject_to_idx = subject_to_idx
        self.eeg_dim = eeg_dim
        self.gaze_dim = gaze_dim
        self.records = None
        if cache_records:
            self.records = [read_record_at(jsonl_path, int(example["offset"])) for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def _record(self, index: int) -> dict[str, Any]:
        if self.records is not None:
            return self.records[index]
        return read_record_at(self.jsonl_path, int(self.examples[index]["offset"]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record(index)
        eeg = np.asarray(record.get("eeg") or [], dtype=np.float32)
        gaze = np.asarray(record.get("gaze") or [], dtype=np.float32)
        if eeg.ndim != 2:
            eeg = np.zeros((0, self.eeg_dim), dtype=np.float32)
        if gaze.ndim != 2:
            gaze = np.zeros((0, self.gaze_dim), dtype=np.float32)

        n_steps = min(len(record.get("words") or []), int(eeg.shape[0]), int(gaze.shape[0]))
        eeg = eeg[:n_steps, : self.eeg_dim]
        gaze = gaze[:n_steps, : self.gaze_dim]

        masks = record.get("masks") or {}
        eeg_missing = np.asarray(masks.get("eeg_missing", [1] * n_steps)[:n_steps], dtype=np.bool_)
        gaze_missing = np.asarray(masks.get("gaze_missing", [1] * n_steps)[:n_steps], dtype=np.bool_)
        if len(eeg_missing) < n_steps:
            eeg_missing = np.pad(eeg_missing, (0, n_steps - len(eeg_missing)), constant_values=True)
        if len(gaze_missing) < n_steps:
            gaze_missing = np.pad(gaze_missing, (0, n_steps - len(gaze_missing)), constant_values=True)

        finite_eeg = np.isfinite(eeg).all(axis=1) if n_steps else np.zeros(0, dtype=np.bool_)
        finite_gaze = np.isfinite(gaze).all(axis=1) if n_steps else np.zeros(0, dtype=np.bool_)
        eeg_missing = eeg_missing | ~finite_eeg
        gaze_missing = gaze_missing | ~finite_gaze
        valid_mask = np.ones(n_steps, dtype=np.bool_)

        subject = str(record["subject"])
        return {
            "eeg_seq": torch.from_numpy(eeg),
            "gaze_seq": torch.from_numpy(gaze),
            "eeg_missing": torch.from_numpy(eeg_missing),
            "gaze_missing": torch.from_numpy(gaze_missing),
            "valid_mask": torch.from_numpy(valid_mask),
            "label": torch.tensor(record_label(record), dtype=torch.long),
            "subject": torch.tensor(self.subject_to_idx[subject], dtype=torch.long),
            "subject_name": subject,
            "sequence_id": record.get("sequence_id", ""),
            "sequence_length": torch.tensor(n_steps, dtype=torch.long),
        }


def collate_nocs(batch: list[dict[str, Any]], max_len: int | None = None) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch.")
    lengths = [int(item["sequence_length"]) for item in batch]
    t_max = max(lengths)
    if max_len is not None:
        t_max = min(t_max, max_len)
    bsz = len(batch)
    eeg_dim = int(batch[0]["eeg_seq"].shape[1])
    gaze_dim = int(batch[0]["gaze_seq"].shape[1])

    eeg = torch.zeros(bsz, t_max, eeg_dim, dtype=torch.float32)
    gaze = torch.zeros(bsz, t_max, gaze_dim, dtype=torch.float32)
    eeg_missing = torch.ones(bsz, t_max, dtype=torch.bool)
    gaze_missing = torch.ones(bsz, t_max, dtype=torch.bool)
    valid_mask = torch.zeros(bsz, t_max, dtype=torch.bool)
    labels = torch.empty(bsz, dtype=torch.long)
    subjects = torch.empty(bsz, dtype=torch.long)
    sequence_lengths = torch.empty(bsz, dtype=torch.long)
    sequence_ids = []
    subject_names = []

    for idx, item in enumerate(batch):
        length = min(int(item["sequence_length"]), t_max)
        if length:
            eeg[idx, :length] = item["eeg_seq"][:length]
            gaze[idx, :length] = item["gaze_seq"][:length]
            eeg_missing[idx, :length] = item["eeg_missing"][:length].bool()
            gaze_missing[idx, :length] = item["gaze_missing"][:length].bool()
            valid_mask[idx, :length] = item["valid_mask"][:length].bool()
        labels[idx] = item["label"]
        subjects[idx] = item["subject"]
        sequence_lengths[idx] = length
        sequence_ids.append(item["sequence_id"])
        subject_names.append(item["subject_name"])

    return {
        "eeg": eeg,
        "gaze": gaze,
        "eeg_missing": eeg_missing,
        "gaze_missing": gaze_missing,
        "valid_mask": valid_mask,
        "labels": labels,
        "subjects": subjects,
        "sequence_lengths": sequence_lengths,
        "sequence_ids": sequence_ids,
        "subject_names": subject_names,
    }


def make_datasets(
    sequence_jsonl: Path,
    split: SplitSubjects,
    cache_records: bool = False,
) -> tuple[NOCSSequenceDataset, NOCSSequenceDataset, NOCSSequenceDataset, dict[str, Any]]:
    all_subjects = sorted(set(split.train + split.val + split.test))
    subject_to_idx = {subject: idx for idx, subject in enumerate(all_subjects)}
    train_examples = collect_examples(sequence_jsonl, split.train, "train")
    val_examples = collect_examples(sequence_jsonl, split.val, "val")
    test_examples = collect_examples(sequence_jsonl, split.test, "test")
    all_examples = train_examples + val_examples + test_examples
    if not train_examples or not val_examples or not test_examples:
        raise RuntimeError(
            "Empty train/val/test split: train=%d val=%d test=%d"
            % (len(train_examples), len(val_examples), len(test_examples))
        )
    eeg_dim, gaze_dim = infer_dims(sequence_jsonl, all_examples)
    train = NOCSSequenceDataset(sequence_jsonl, train_examples, subject_to_idx, eeg_dim, gaze_dim, cache_records)
    val = NOCSSequenceDataset(sequence_jsonl, val_examples, subject_to_idx, eeg_dim, gaze_dim, cache_records)
    test = NOCSSequenceDataset(sequence_jsonl, test_examples, subject_to_idx, eeg_dim, gaze_dim, cache_records)
    meta = {
        "subject_to_idx": subject_to_idx,
        "eeg_dim": eeg_dim,
        "gaze_dim": gaze_dim,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "n_test": len(test_examples),
    }
    return train, val, test, meta


def make_loader(
    dataset: NOCSSequenceDataset,
    batch_size: int,
    shuffle: bool,
    max_len: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_nocs(batch, max_len=max_len),
    )
