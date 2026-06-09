"""Subject split manifests for AdaGTCN-aligned experiments.

The original AdaGTCN comparison is commonly described as 12 train, 2
validation, and 4 test subjects. That requires 18 labeled subjects. This
workspace currently has 16 labeled Y subjects with both NR and TSR raw files,
so the honest available-Y split is 12/2/2, plus LOSO for stability.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


Y16_SUBJECTS = [
    "YAC",
    "YAG",
    "YAK",
    "YDG",
    "YDR",
    "YFR",
    "YFS",
    "YHS",
    "YIS",
    "YLS",
    "YMD",
    "YRK",
    "YRP",
    "YSD",
    "YSL",
    "YTL",
]

TASKS = ("NR", "TSR")
MAT_FILE_RE = re.compile(r"^results(?P<subject>[A-Z0-9]+)_(?P<task>NR|TSR)\.mat$")


@dataclass(frozen=True)
class SubjectSplit:
    protocol: str
    seed: int | None
    train: list[str]
    val: list[str]
    test: list[str]
    note: str

    def flat_rows(self) -> list[dict[str, object]]:
        rows = []
        for role in ("train", "val", "test"):
            for subject in getattr(self, role):
                rows.append(
                    {
                        "protocol": self.protocol,
                        "seed": self.seed,
                        "role": role,
                        "subject": subject,
                        "n_train": len(self.train),
                        "n_val": len(self.val),
                        "n_test": len(self.test),
                        "note": self.note,
                    }
                )
        return rows


def discover_raw_subjects(data_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return subjects that have both NR and TSR raw Matlab files."""
    inventory: dict[str, set[str]] = {}
    for path in data_dir.glob("results*.mat"):
        match = MAT_FILE_RE.match(path.name)
        if not match:
            continue
        inventory.setdefault(match.group("subject"), set()).add(match.group("task"))

    complete = sorted([s for s, tasks in inventory.items() if set(TASKS).issubset(tasks)])
    return complete, {s: sorted(tasks) for s, tasks in sorted(inventory.items())}


def make_y16_12_2_2(seed: int, subjects: Iterable[str] = Y16_SUBJECTS) -> SubjectSplit:
    subjects = list(subjects)
    if len(subjects) != 16:
        raise ValueError(f"Y16_12_2_2 requires 16 subjects, got {len(subjects)}")

    shuffled = subjects.copy()
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:12]
    val = shuffled[12:14]
    test = shuffled[14:]
    return SubjectSplit(
        protocol=f"Y16_12_2_2_seed{seed}",
        seed=seed,
        train=train,
        val=val,
        test=test,
        note=(
            "AdaGTCN-style subject-independent split on the available 16 labeled "
            "Y subjects. This is not the original 18-subject 12/2/4 split."
        ),
    )


def make_y16_loso(subjects: Iterable[str] = Y16_SUBJECTS) -> list[SubjectSplit]:
    subjects = sorted(subjects)
    folds = []
    n_subjects = len(subjects)
    for idx, held_out in enumerate(subjects):
        val_subject = None
        for offset in range(1, n_subjects):
            candidate = subjects[(idx + offset) % n_subjects]
            if candidate != held_out:
                val_subject = candidate
                break
        if val_subject is None:
            raise ValueError("LOSO requires at least two subjects.")

        train = [s for s in subjects if s not in {held_out, val_subject}]
        folds.append(
            SubjectSplit(
                protocol=f"Y16_LOSO_{held_out}",
                seed=None,
                train=train,
                val=[val_subject],
                test=[held_out],
                note=(
                    "Leave-one-subject-out on the available 16 labeled Y subjects. "
                    "LOSO with deterministic inner validation subject; no test-as-validation."
                ),
            )
        )
    return folds


def build_manifest(data_dir: Path, seed: int) -> dict[str, object]:
    complete_subjects, inventory = discover_raw_subjects(data_dir)
    available_y = [s for s in Y16_SUBJECTS if s in complete_subjects]
    missing_y = [s for s in Y16_SUBJECTS if s not in complete_subjects]

    splits: list[SubjectSplit] = []
    if len(available_y) == 16:
        splits.append(make_y16_12_2_2(seed, available_y))
        splits.extend(make_y16_loso(available_y))

    return {
        "data_dir": str(data_dir),
        "available_complete_subjects": complete_subjects,
        "available_y16_subjects": available_y,
        "missing_y16_subjects": missing_y,
        "inventory": inventory,
        "strict_adagtcn_12_2_4_possible": len(available_y) >= 18,
        "recommended_primary_protocol": f"Y16_12_2_2_seed{seed}",
        "recommended_stability_protocol": "Y16_LOSO",
        "splits": [asdict(split) for split in splits],
    }


def write_manifest(manifest: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "subject_splits.json"
    csv_path = output_dir / "subject_splits.csv"
    readme_path = output_dir / "alignment_protocol_note.md"

    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    splits = [
        SubjectSplit(
            protocol=s["protocol"],
            seed=s["seed"],
            train=list(s["train"]),
            val=list(s["val"]),
            test=list(s["test"]),
            note=s["note"],
        )
        for s in manifest["splits"]
    ]
    rows = [row for split in splits for row in split.flat_rows()]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["protocol", "seed", "role", "subject", "n_train", "n_val", "n_test", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    note = [
        "# AdaGTCN Alignment Protocol\n",
        "\n",
        "## Decision\n",
        "\n",
        "- Use `Y16_12_2_2_seed0` as the primary available-Y subject split.\n",
        "- Use `Y16_LOSO_*` folds as the stability protocol.\n",
        "- Do not call the available-Y split `12/2/4`: only 16 labeled Y subjects are present.\n",
        "\n",
        "## Why\n",
        "\n",
        "The AdaGTCN paper-style 12 train / 2 validation / 4 test protocol needs 18 labeled subjects. "
        "This workspace has 16 labeled Y subjects with both NR and TSR raw Matlab files, so an exact "
        "18-subject split cannot be created without additional labeled subjects or the original "
        "AdaGTCN subject list.\n",
        "\n",
        "## Input Alignment\n",
        "\n",
        "Protocol alignment alone is not enough. AdaGTCN-style comparison must use word/fixation-level "
        "EEG and eye-tracking sequences from the raw `.mat` files rather than only sentence-level "
        "precomputed feature vectors.\n",
    ]
    readme_path.write_text("".join(note), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AdaGTCN-aligned subject split manifests.")
    parser.add_argument("--data-dir", default="data/train", type=Path)
    parser.add_argument("--output-dir", default="reports/adagtcn_aligned", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir, args.seed)
    write_manifest(manifest, args.output_dir)
    print(f"Wrote split manifest to {args.output_dir}")
    print(f"Available Y subjects: {len(manifest['available_y16_subjects'])}")
    print(f"Strict 18-subject 12/2/4 possible: {manifest['strict_adagtcn_12_2_4_possible']}")
    print(f"Recommended primary protocol: {manifest['recommended_primary_protocol']}")


if __name__ == "__main__":
    main()
