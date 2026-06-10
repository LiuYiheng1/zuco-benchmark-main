"""Extract ZuCo word-level EEG band + gaze sequences from Matlab v7.3 files.

This extractor is read-only with respect to the original .mat files. It emits
sentence-level JSONL records for AdaGTCN-style word/fixation-level experiments,
using word-level EEG band representations and gaze features only. Raw EEG and
raw eye-tracking matrices are inspected for shape statistics but are not written
to the JSONL output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MAT_FILE_RE = re.compile(r"^results(?P<subject>[A-Z0-9]+)_(?P<task>NR|TSR)\.mat$")
LABEL_BY_TASK = {"NR": 0, "TSR": 1}

EEG_FIELDS = ["TRT_t1", "TRT_t2", "TRT_a1", "TRT_a2", "TRT_b1", "TRT_b2", "TRT_g1", "TRT_g2"]
EEG_CHANNELS = 105
EEG_DIM = len(EEG_FIELDS) * EEG_CHANNELS

GAZE_SCALAR_FIELDS = ["FFD", "GD", "GPT", "TRT", "SFD", "nFixations", "meanPupilSize"]
GAZE_POSITION_STATS = ["firstFixPosition", "lastFixPosition", "meanFixPosition"]
GAZE_DIM = len(GAZE_SCALAR_FIELDS) + len(GAZE_POSITION_STATS)


def mat_index(dataset: h5py.Dataset, idx: int) -> Any:
    """Index Matlab cell vectors saved as either Nx1 or 1xN object datasets."""
    if dataset.ndim == 2 and dataset.shape[1] == 1:
        return dataset[idx, 0]
    if dataset.ndim == 2 and dataset.shape[0] == 1:
        return dataset[0, idx]
    return dataset[idx]


def is_null_placeholder_array(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return True
    if arr.dtype == object:
        return False
    if arr.shape == (2,) and np.all(arr == 0):
        return True
    return bool(arr.size <= 2 and np.all(arr == 0))


def deref(file: h5py.File, obj: Any, max_depth: int = 8) -> h5py.Dataset | h5py.Group | None:
    """Safely dereference HDF5 object references and nested Matlab cells."""
    cur = obj
    for _ in range(max_depth):
        if isinstance(cur, (h5py.Dataset, h5py.Group)):
            if isinstance(cur, h5py.Dataset) and cur.dtype == object:
                try:
                    arr = np.asarray(cur[()])
                except Exception:
                    return cur
                if is_null_placeholder_array(arr):
                    return None
                next_obj = None
                for ref in arr.reshape(-1):
                    try:
                        next_obj = file[ref]
                        break
                    except Exception:
                        continue
                if next_obj is None:
                    return None
                cur = next_obj
                continue
            return cur
        try:
            cur = file[cur]
        except Exception:
            return None
    return None


def dataset_array(file: h5py.File, obj: Any) -> np.ndarray | None:
    leaf = deref(file, obj)
    if not isinstance(leaf, h5py.Dataset):
        return None
    try:
        arr = np.asarray(leaf[()])
    except Exception:
        return None
    if is_null_placeholder_array(arr):
        return None
    return arr


def matlab_string(file: h5py.File, obj: Any) -> str:
    arr = dataset_array(file, obj)
    if arr is None:
        return ""
    chars: list[str] = []
    for value in arr.reshape(-1):
        try:
            code = int(value)
        except Exception:
            continue
        if code > 0:
            chars.append(chr(code))
    return "".join(chars)


def numeric_array(file: h5py.File, obj: Any) -> np.ndarray | None:
    arr = dataset_array(file, obj)
    if arr is None or arr.dtype == object:
        return None
    try:
        out = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None
    if out.size == 0 or not np.isfinite(out).any():
        return None
    return out


def numeric_scalar(file: h5py.File, obj: Any) -> float | None:
    arr = numeric_array(file, obj)
    if arr is None:
        return None
    flat = arr.reshape(-1)
    if flat.size == 0:
        return None
    value = float(flat[0])
    if not math.isfinite(value):
        return None
    return value


def numeric_vector(file: h5py.File, obj: Any) -> list[float]:
    arr = numeric_array(file, obj)
    if arr is None:
        return []
    return [float(x) for x in arr.reshape(-1) if math.isfinite(float(x))]


def object_shape(file: h5py.File, obj: Any) -> list[int] | None:
    leaf = deref(file, obj)
    if not isinstance(leaf, h5py.Dataset):
        return None
    arr = dataset_array(file, leaf)
    if arr is None:
        return None
    return [int(dim) for dim in arr.shape]


def vector105(file: h5py.File, obj: Any) -> tuple[list[float], bool]:
    arr = numeric_array(file, obj)
    if arr is None:
        return [0.0] * EEG_CHANNELS, False
    flat = arr.reshape(-1)
    if flat.size < EEG_CHANNELS:
        return [0.0] * EEG_CHANNELS, False
    values = flat[:EEG_CHANNELS].astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        return [0.0] * EEG_CHANNELS, False
    return [float(x) for x in values], True


def read_eeg_vector(file: h5py.File, word_group: h5py.Group, word_idx: int) -> tuple[list[float], int]:
    values: list[float] = []
    missing_fields = 0
    for field in EEG_FIELDS:
        if field not in word_group:
            values.extend([0.0] * EEG_CHANNELS)
            missing_fields += 1
            continue
        field_values, ok = vector105(file, mat_index(word_group[field], word_idx))
        values.extend(field_values)
        missing_fields += int(not ok)
    return values, int(missing_fields > 0)


def read_gaze_vector(file: h5py.File, word_group: h5py.Group, word_idx: int) -> tuple[list[float], int, int]:
    scalars: dict[str, float | None] = {}
    for field in GAZE_SCALAR_FIELDS:
        scalars[field] = numeric_scalar(file, mat_index(word_group[field], word_idx)) if field in word_group else None

    nfix = scalars.get("nFixations")
    nfix_zero = int(nfix is None or nfix <= 0)
    gaze_missing = int(nfix_zero)

    values = [float(scalars[field]) if scalars[field] is not None else 0.0 for field in GAZE_SCALAR_FIELDS]

    positions = numeric_vector(file, mat_index(word_group["fixPositions"], word_idx)) if "fixPositions" in word_group else []
    valid_positions = [x for x in positions if math.isfinite(x) and x > 0]
    if valid_positions:
        values.extend([float(valid_positions[0]), float(valid_positions[-1]), float(np.mean(valid_positions))])
    else:
        values.extend([0.0, 0.0, 0.0])

    if any(v > 0 for v in values[:6]):
        gaze_missing = 0
    return values, gaze_missing, nfix_zero


def read_wordbounds(file: h5py.File, sentence_data: h5py.Group, sent_idx: int) -> list[list[float]]:
    if "wordbounds" not in sentence_data:
        return []
    arr = numeric_array(file, mat_index(sentence_data["wordbounds"], sent_idx))
    if arr is None:
        return []
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 4:
        arr = arr.T
    return [[float(x) for x in row] for row in arr]


def read_omission_rate(file: h5py.File, sentence_data: h5py.Group, sent_idx: int) -> float | None:
    if "omissionRate" not in sentence_data:
        return None
    return numeric_scalar(file, mat_index(sentence_data["omissionRate"], sent_idx))


def sentence_count(sentence_data: h5py.Group) -> int:
    refs = sentence_data["word"]
    if refs.ndim == 2 and refs.shape[1] == 1:
        return int(refs.shape[0])
    if refs.ndim == 2 and refs.shape[0] == 1:
        return int(refs.shape[1])
    return int(refs.shape[0])


def word_count(word_group: h5py.Group) -> int:
    content = word_group["content"]
    if content.ndim == 2 and content.shape[1] == 1:
        return int(content.shape[0])
    if content.ndim == 2 and content.shape[0] == 1:
        return int(content.shape[1])
    return int(content.shape[0])


def update_shape_counter(counter: Counter[str], shape: list[int] | None) -> None:
    if shape is not None:
        counter["x".join(str(x) for x in shape)] += 1


def extract_file(mat_path: Path, out_f: Any, preview: list[str]) -> dict[str, Any]:
    match = MAT_FILE_RE.match(mat_path.name)
    if not match:
        raise ValueError(f"Unexpected file name: {mat_path.name}")
    subject = match.group("subject")
    task = match.group("task")
    label = LABEL_BY_TASK[task]

    stats: dict[str, Any] = {
        "source_file": mat_path.name,
        "subject": subject,
        "task": task,
        "n_sentences": 0,
        "n_words": 0,
        "missing_eeg_word_count": 0,
        "missing_gaze_word_count": 0,
        "nfix_zero_word_count": 0,
        "sequence_lengths": [],
        "rawEEG_shapes": Counter(),
        "rawET_shapes": Counter(),
    }

    with h5py.File(mat_path, "r") as file:
        sentence_data = file["sentenceData"]
        n_sentences = sentence_count(sentence_data)

        for sent_idx in range(n_sentences):
            word_group = deref(file, mat_index(sentence_data["word"], sent_idx))
            if not isinstance(word_group, h5py.Group) or "content" not in word_group:
                continue

            words: list[str] = []
            eeg_rows: list[list[float]] = []
            gaze_rows: list[list[float]] = []
            eeg_missing: list[int] = []
            gaze_missing: list[int] = []
            nfix_zero: list[int] = []
            n_words = word_count(word_group)

            for word_idx in range(n_words):
                words.append(matlab_string(file, mat_index(word_group["content"], word_idx)))

                eeg_vec, eeg_miss = read_eeg_vector(file, word_group, word_idx)
                gaze_vec, gaze_miss, zero_fix = read_gaze_vector(file, word_group, word_idx)
                eeg_rows.append(eeg_vec)
                gaze_rows.append(gaze_vec)
                eeg_missing.append(eeg_miss)
                gaze_missing.append(gaze_miss)
                nfix_zero.append(zero_fix)

                if "rawEEG" in word_group:
                    update_shape_counter(stats["rawEEG_shapes"], object_shape(file, mat_index(word_group["rawEEG"], word_idx)))
                if "rawET" in word_group:
                    update_shape_counter(stats["rawET_shapes"], object_shape(file, mat_index(word_group["rawET"], word_idx)))

            sentence_content = matlab_string(file, mat_index(sentence_data["content"], sent_idx)) if "content" in sentence_data else ""
            omission_rate = read_omission_rate(file, sentence_data, sent_idx)
            record = {
                "subject": subject,
                "task": task,
                "label": label,
                "sentence_id": sent_idx,
                "sentence_content": sentence_content,
                "n_words": n_words,
                "words": words,
                "eeg": eeg_rows,
                "gaze": gaze_rows,
                "masks": {
                    "eeg_missing": eeg_missing,
                    "gaze_missing": gaze_missing,
                    "nfix_zero": nfix_zero,
                },
                "metadata": {
                    "wordbounds": read_wordbounds(file, sentence_data, sent_idx),
                    "omissionRate": omission_rate,
                    "source_file": mat_path.name,
                    "eeg_fields": EEG_FIELDS,
                    "gaze_fields": GAZE_SCALAR_FIELDS + GAZE_POSITION_STATS,
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

            stats["n_sentences"] += 1
            stats["n_words"] += n_words
            stats["missing_eeg_word_count"] += sum(eeg_missing)
            stats["missing_gaze_word_count"] += sum(gaze_missing)
            stats["nfix_zero_word_count"] += sum(nfix_zero)
            stats["sequence_lengths"].append(n_words)

            if len(preview) < 3:
                preview.append(
                    f"{subject}/{task}/sentence_id={sent_idx} "
                    f"n_words={n_words} eeg_dim={len(eeg_rows[0]) if eeg_rows else EEG_DIM} "
                    f"gaze_dim={len(gaze_rows[0]) if gaze_rows else GAZE_DIM} "
                    f"missing_eeg={sum(eeg_missing)} missing_gaze={sum(gaze_missing)} "
                    f"nfix_zero={sum(nfix_zero)}"
                )

    lengths = stats.pop("sequence_lengths")
    stats["average_sequence_length"] = float(np.mean(lengths)) if lengths else 0.0
    stats["max_sequence_length"] = int(max(lengths)) if lengths else 0
    stats["rawEEG_shapes"] = dict(stats["rawEEG_shapes"])
    stats["rawET_shapes"] = dict(stats["rawET_shapes"])
    return stats


def list_input_files(data_dir: Path, subjects: list[str] | None, max_files: int | None) -> list[Path]:
    files = sorted(path for path in data_dir.glob("results*_*.mat") if MAT_FILE_RE.match(path.name))
    if subjects:
        wanted = set(subjects)
        files = [path for path in files if MAT_FILE_RE.match(path.name).group("subject") in wanted]  # type: ignore[union-attr]
    if max_files is not None:
        files = files[:max_files]
    return files


def build_summary(files: list[Path], per_file: list[dict[str, Any]]) -> dict[str, Any]:
    label_distribution = Counter()
    subject_distribution = Counter()
    subject_task: dict[str, dict[str, Any]] = {}
    total_sequences = 0
    total_words = 0

    for item in per_file:
        label_distribution[str(LABEL_BY_TASK[item["task"]])] += int(item["n_sentences"])
        subject_distribution[item["subject"]] += int(item["n_sentences"])
        total_sequences += int(item["n_sentences"])
        total_words += int(item["n_words"])
        subject_task[f"{item['subject']}_{item['task']}"] = item

    return {
        "input_files": [path.name for path in files],
        "n_files": len(files),
        "total_sequences": total_sequences,
        "total_words": total_words,
        "eeg_dim": EEG_DIM,
        "eeg_fields": EEG_FIELDS,
        "gaze_dim": GAZE_DIM,
        "gaze_fields": GAZE_SCALAR_FIELDS + GAZE_POSITION_STATS,
        "label_mapping": LABEL_BY_TASK,
        "label_distribution": dict(label_distribution),
        "subject_distribution": dict(subject_distribution),
        "subject_task_stats": subject_task,
        "notes": [
            "rawEEG/rawET shapes are counted but raw matrices are not written to JSONL.",
            "TRT_*/FFD_*/GD_*/GPT_*/SFD_* features are word-level EEG band representations, not raw EEG waveform.",
            "No explicit allFixations onset/offset field is assumed; rawET first column is only treated as a timestamp-like value.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ZuCo word-level EEG band + gaze sequence JSONL.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/adagtcn_aligned/zuco_word_band_gaze_sequences.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/adagtcn_aligned/zuco_word_band_extraction_summary.json"))
    parser.add_argument("--preview-txt", type=Path, default=Path("reports/adagtcn_aligned/zuco_word_band_preview.txt"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    files = list_input_files(args.data_dir, args.subjects, args.max_files)
    if not files:
        raise SystemExit(f"No matching .mat files found under {args.data_dir}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.preview_txt.parent.mkdir(parents=True, exist_ok=True)

    per_file: list[dict[str, Any]] = []
    preview: list[str] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for path in files:
            print(f"Extracting {path.name}", flush=True)
            per_file.append(extract_file(path, out_f, preview))

    summary = build_summary(files, per_file)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    args.preview_txt.write_text("\n".join(preview) + "\n", encoding="utf-8")

    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Wrote summary: {args.summary_json}")
    print(f"Wrote preview: {args.preview_txt}")
    print(f"Sequences={summary['total_sequences']} words={summary['total_words']} eeg_dim={EEG_DIM} gaze_dim={GAZE_DIM}")


if __name__ == "__main__":
    main()
