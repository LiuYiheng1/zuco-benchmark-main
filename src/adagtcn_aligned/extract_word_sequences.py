"""Extract word/fixation-level ZuCo sequences for AdaGTCN-aligned models.

This is intentionally separate from the historical sentence-level benchmark
scripts. It reads the official HDF5 Matlab files and emits sequence JSONL plus
a CSV manifest. EEG extraction can be expensive because each word-level EEG
field is stored behind Matlab object references, so use --max-subjects,
--max-sentences, --max-words, and --eeg-fields for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MAT_FILE_RE = re.compile(r"^results(?P<subject>[A-Z0-9]+)_(?P<task>NR|TSR)\.mat$")
LABEL_TO_Y = {"NR": 1, "TSR": 0}

GAZE_FIELDS = ["FFD", "GD", "GPT", "TRT", "nFixations", "meanPupilSize"]
FIX_VECTOR_FIELDS = ["fixPositions"]
DEFAULT_EEG_FIELDS = ["TRT_t1", "TRT_t2", "TRT_a1", "TRT_a2", "TRT_b1", "TRT_b2", "TRT_g1", "TRT_g2"]


def matlab_ref(dataset: h5py.Dataset, idx: int) -> Any:
    if dataset.ndim == 2 and dataset.shape[1] == 1:
        return dataset[idx, 0]
    if dataset.ndim == 2 and dataset.shape[0] == 1:
        return dataset[0, idx]
    return dataset[idx]


def deref(file: h5py.File, ref: Any) -> h5py.Dataset | h5py.Group | None:
    try:
        return file[ref]
    except Exception:
        return None


def matlab_string(file: h5py.File, ref: Any) -> str:
    obj = deref(file, ref)
    if obj is None:
        return ""
    arr = np.asarray(obj).reshape(-1)
    chars = []
    for value in arr:
        try:
            chars.append(chr(int(value)))
        except Exception:
            pass
    return "".join(chars)


def numeric_array(file: h5py.File, ref: Any) -> np.ndarray | None:
    obj = deref(file, ref)
    if obj is None or not isinstance(obj, h5py.Dataset):
        return None
    try:
        arr = np.asarray(obj, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def numeric_scalar(file: h5py.File, ref: Any) -> float | None:
    arr = numeric_array(file, ref)
    if arr is None:
        return None
    value = float(arr.reshape(-1)[0])
    if math.isnan(value):
        return None
    return value


def numeric_vector(file: h5py.File, ref: Any) -> list[float]:
    arr = numeric_array(file, ref)
    if arr is None:
        return []
    values = []
    for value in arr.reshape(-1):
        value = float(value)
        if not math.isnan(value):
            values.append(value)
    return values


def read_eeg_features(
    file: h5py.File,
    word_group: h5py.Group,
    word_idx: int,
    eeg_mode: str,
    eeg_fields: list[str],
) -> tuple[list[float], list[str]]:
    if eeg_mode == "none":
        return [], []

    values: list[float] = []
    valid_fields: list[str] = []
    for field in eeg_fields:
        if field not in word_group:
            continue
        ref = matlab_ref(word_group[field], word_idx)
        arr = numeric_array(file, ref)
        if arr is None:
            continue
        arr = arr.astype(np.float64, copy=False).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        valid_fields.append(field)
        if eeg_mode == "band_means":
            values.append(float(np.nanmean(arr)))
        elif eeg_mode == "band_vectors":
            values.extend([float(x) for x in arr[:105]])
        else:
            raise ValueError(f"Unsupported eeg_mode: {eeg_mode}")
    return values, valid_fields


def has_positive_fixation(gaze: dict[str, Any]) -> bool:
    for field in ("FFD", "GD", "GPT", "TRT", "nFixations"):
        value = gaze.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            return True
    return False


def sequence_id(subject: str, task: str, sentence_idx: int) -> str:
    return f"{subject}_{task}_{sentence_idx}"


def extract_mat_file(
    mat_path: Path,
    sequence_writer: Any,
    max_sentences: int | None,
    max_words: int | None,
    eeg_mode: str,
    eeg_fields: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    match = MAT_FILE_RE.match(mat_path.name)
    if not match:
        raise ValueError(f"Unexpected Matlab filename: {mat_path.name}")
    subject = match.group("subject")
    task = match.group("task")
    y = LABEL_TO_Y[task]

    manifest_rows: list[dict[str, Any]] = []
    audit = {
        "file": str(mat_path),
        "subject": subject,
        "task": task,
        "n_sequences": 0,
        "n_words": 0,
        "n_words_with_gaze": 0,
        "n_words_with_eeg": 0,
    }

    with h5py.File(mat_path, "r") as file:
        sentence_data = file["sentenceData"]
        word_refs = sentence_data["word"]
        content_refs = sentence_data.get("content")
        n_sentences = word_refs.shape[0] if word_refs.ndim == 2 and word_refs.shape[1] == 1 else word_refs.shape[1]
        if max_sentences is not None:
            n_sentences = min(n_sentences, max_sentences)

        for sent_idx in range(n_sentences):
            word_group = deref(file, matlab_ref(word_refs, sent_idx))
            if word_group is None or not isinstance(word_group, h5py.Group) or "content" not in word_group:
                continue

            sentence_text = ""
            if content_refs is not None:
                sentence_text = matlab_string(file, matlab_ref(content_refs, sent_idx))

            content = word_group["content"]
            n_words = content.shape[0] if content.ndim == 2 and content.shape[1] == 1 else content.shape[1]
            if max_words is not None:
                n_words = min(n_words, max_words)

            words = []
            gaze_rows = []
            gaze_masks = []
            eeg_rows = []
            eeg_masks = []
            eeg_valid_fields = []

            for word_idx in range(n_words):
                word_text = matlab_string(file, matlab_ref(content, word_idx))
                gaze = {}
                gaze_valid = False
                for field in GAZE_FIELDS:
                    if field in word_group:
                        value = numeric_scalar(file, matlab_ref(word_group[field], word_idx))
                        gaze[field] = value
                        gaze_valid = gaze_valid or value is not None
                    else:
                        gaze[field] = None
                for field in FIX_VECTOR_FIELDS:
                    if field in word_group:
                        gaze[field] = numeric_vector(file, matlab_ref(word_group[field], word_idx))
                    else:
                        gaze[field] = []

                gaze_valid = has_positive_fixation(gaze)
                if gaze_valid:
                    eeg_values, valid_fields = read_eeg_features(file, word_group, word_idx, eeg_mode, eeg_fields)
                else:
                    eeg_values, valid_fields = [], []
                has_eeg = len(eeg_values) > 0

                words.append(word_text)
                gaze_rows.append(gaze)
                gaze_masks.append(gaze_valid)
                eeg_rows.append(eeg_values)
                eeg_masks.append(has_eeg)
                eeg_valid_fields.append(valid_fields)

                audit["n_words"] += 1
                audit["n_words_with_gaze"] += int(gaze_valid)
                audit["n_words_with_eeg"] += int(has_eeg)

            record = {
                "sequence_id": sequence_id(subject, task, sent_idx),
                "subject": subject,
                "task": task,
                "y": y,
                "sentence_idx": sent_idx,
                "sentence_text": sentence_text,
                "words": words,
                "gaze": gaze_rows,
                "gaze_mask": gaze_masks,
                "eeg_mode": eeg_mode,
                "eeg_fields": eeg_fields,
                "eeg": eeg_rows,
                "eeg_mask": eeg_masks,
                "eeg_valid_fields": eeg_valid_fields,
            }
            sequence_writer.write(json.dumps(record, ensure_ascii=True) + "\n")

            manifest_rows.append(
                {
                    "sequence_id": record["sequence_id"],
                    "subject": subject,
                    "task": task,
                    "y": y,
                    "sentence_idx": sent_idx,
                    "n_words": len(words),
                    "n_words_with_gaze": sum(gaze_masks),
                    "n_words_with_eeg": sum(eeg_masks),
                    "eeg_mode": eeg_mode,
                    "eeg_dim_first_nonempty": next((len(row) for row in eeg_rows if row), 0),
                }
            )
            audit["n_sequences"] += 1

    return manifest_rows, audit


def list_mat_files(data_dir: Path, subjects: list[str] | None, max_subjects: int | None) -> list[Path]:
    files = sorted(data_dir.glob("resultsY*_*.mat"))
    if subjects:
        subject_set = set(subjects)
        filtered = []
        for path in files:
            match = MAT_FILE_RE.match(path.name)
            if match and match.group("subject") in subject_set:
                filtered.append(path)
        files = filtered
    if max_subjects is not None:
        seen = []
        keep = []
        for path in files:
            match = MAT_FILE_RE.match(path.name)
            if not match:
                continue
            subject = match.group("subject")
            if subject not in seen:
                seen.append(subject)
            if len(seen) <= max_subjects:
                keep.append(path)
        files = keep
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AdaGTCN-aligned word/fixation sequences.")
    parser.add_argument("--data-dir", default="data/train", type=Path)
    parser.add_argument("--output-dir", default="data/adagtcn_aligned", type=Path)
    parser.add_argument("--prefix", default="y16_word_sequences")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--eeg-mode", choices=["none", "band_means", "band_vectors"], default="none")
    parser.add_argument("--eeg-fields", default=",".join(DEFAULT_EEG_FIELDS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seq_path = args.output_dir / f"{args.prefix}_sequences.jsonl"
    manifest_path = args.output_dir / f"{args.prefix}_manifest.csv"
    audit_path = args.output_dir / f"{args.prefix}_audit.json"

    eeg_fields = [field.strip() for field in args.eeg_fields.split(",") if field.strip()]
    files = list_mat_files(args.data_dir, args.subjects, args.max_subjects)
    all_manifest_rows: list[dict[str, Any]] = []
    audits = []

    with seq_path.open("w", encoding="utf-8") as seq_f:
        for mat_path in files:
            print(f"Extracting {mat_path.name}", flush=True)
            rows, audit = extract_mat_file(
                mat_path=mat_path,
                sequence_writer=seq_f,
                max_sentences=args.max_sentences,
                max_words=args.max_words,
                eeg_mode=args.eeg_mode,
                eeg_fields=eeg_fields,
            )
            all_manifest_rows.extend(rows)
            audits.append(audit)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sequence_id",
            "subject",
            "task",
            "y",
            "sentence_idx",
            "n_words",
            "n_words_with_gaze",
            "n_words_with_eeg",
            "eeg_mode",
            "eeg_dim_first_nonempty",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    summary = {
        "n_files": len(files),
        "n_sequences": sum(a["n_sequences"] for a in audits),
        "n_words": sum(a["n_words"] for a in audits),
        "n_words_with_gaze": sum(a["n_words_with_gaze"] for a in audits),
        "n_words_with_eeg": sum(a["n_words_with_eeg"] for a in audits),
        "eeg_mode": args.eeg_mode,
        "eeg_fields": eeg_fields,
        "files": audits,
    }
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote sequences: {seq_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote audit: {audit_path}")
    print(f"Sequences: {summary['n_sequences']}; words: {summary['n_words']}")
    print(f"Words with gaze: {summary['n_words_with_gaze']}; words with EEG: {summary['n_words_with_eeg']}")


if __name__ == "__main__":
    main()
