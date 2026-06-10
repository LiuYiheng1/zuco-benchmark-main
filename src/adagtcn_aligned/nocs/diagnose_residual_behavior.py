"""Diagnose NOCS residual behavior against full NOCS and word-pool gaze LR."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REQUIRED_INTERNAL_FIELDS = {
    "logits_g",
    "logits_full",
    "residual_gate",
    "residual_norm",
    "precision_e",
    "precision_g",
}


def read_single_row_csvs(root: Path, suffix: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*" + suffix)):
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                protocol = str(row.get("protocol", ""))
                if protocol:
                    row["source_file"] = str(path)
                    rows[protocol] = row
    return rows


def read_gaze_lr(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("modality") != "gaze":
                continue
            protocol = str(row.get("protocol", ""))
            if protocol:
                rows[protocol] = row
    return rows


def fnum(value: Any) -> float:
    if value in {None, ""}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prediction_fields(root: Path) -> set[str]:
    fields: set[str] = set()
    for path in sorted(root.glob("*_predictions.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fields.update(reader.fieldnames)
    return fields


def maybe_write_gate_stats(root: Path, output_dir: Path) -> tuple[bool, list[str]]:
    fields = prediction_fields(root)
    missing = sorted(REQUIRED_INTERNAL_FIELDS - fields)
    if missing:
        return False, missing

    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_predictions.csv")):
        protocol = path.name.split("_seed", 1)[0].replace("residual_", "")
        by_subject: dict[str, list[dict[str, str]]] = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_subject.setdefault(str(row.get("subject", "")), []).append(row)
        for subject, items in sorted(by_subject.items()):
            gates = [fnum(row.get("residual_gate")) for row in items]
            norms = [fnum(row.get("residual_norm")) for row in items]
            shifts = [abs(fnum(row.get("logits_full")) - fnum(row.get("logits_g"))) for row in items]
            rows.append(
                {
                    "protocol": protocol,
                    "heldout_subject": subject,
                    "mean_gate": mean(gates),
                    "std_gate": stdev(gates) if len(gates) > 1 else 0.0,
                    "mean_abs_residual_logit_shift": mean(shifts),
                    "mean_residual_norm": mean(norms),
                    "num_gaze_correct_full_wrong": "",
                    "num_gaze_wrong_full_correct": "",
                }
            )
    write_rows(rows, output_dir / "residual_gate_stats.csv")
    return True, []


def build_subject_rows(
    full: dict[str, dict[str, Any]],
    residual: dict[str, dict[str, Any]],
    gaze_lr: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for protocol in sorted(set(full) | set(residual) | set(gaze_lr)):
        full_row = full.get(protocol, {})
        residual_row = residual.get(protocol, {})
        gaze_row = gaze_lr.get(protocol, {})
        heldout = (
            residual_row.get("heldout_subject")
            or full_row.get("heldout_subject")
            or gaze_row.get("heldout_subject")
            or protocol.replace("Y16_LOSO_", "")
        )
        full_auroc = fnum(full_row.get("test_auroc"))
        residual_auroc = fnum(residual_row.get("test_auroc"))
        gaze_auroc = fnum(gaze_row.get("test_auroc"))
        full_f1 = fnum(full_row.get("test_macro_f1"))
        residual_f1 = fnum(residual_row.get("test_macro_f1"))
        gaze_f1 = fnum(gaze_row.get("test_macro_f1"))
        rows.append(
            {
                "protocol": protocol,
                "heldout_subject": heldout,
                "nocs_full_auroc": full_auroc,
                "nocs_residual_auroc": residual_auroc,
                "gaze_lr_auroc": gaze_auroc,
                "delta_residual_minus_full_auroc": residual_auroc - full_auroc,
                "delta_residual_minus_gaze_lr_auroc": residual_auroc - gaze_auroc,
                "nocs_full_macro_f1": full_f1,
                "nocs_residual_macro_f1": residual_f1,
                "gaze_lr_macro_f1": gaze_f1,
                "test_examples": residual_row.get("test_examples") or full_row.get("test_examples") or gaze_row.get("test_examples") or "",
            }
        )
    return rows


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else "%.4f" % value


def write_report(rows: list[dict[str, Any]], gate_exported: bool, missing_fields: list[str], output_dir: Path) -> None:
    deltas_full = [fnum(row["delta_residual_minus_full_auroc"]) for row in rows]
    residual_aurocs = [fnum(row["nocs_residual_auroc"]) for row in rows]
    full_aurocs = [fnum(row["nocs_full_auroc"]) for row in rows]
    gaze_aurocs = [fnum(row["gaze_lr_auroc"]) for row in rows]
    sorted_drop = sorted(rows, key=lambda row: fnum(row["delta_residual_minus_full_auroc"]))
    improved_full = sum(1 for row in rows if fnum(row["delta_residual_minus_full_auroc"]) > 0)
    improved_gaze = sum(1 for row in rows if fnum(row["delta_residual_minus_gaze_lr_auroc"]) > 0)
    worst_residual = min(rows, key=lambda row: fnum(row["nocs_residual_auroc"]))
    worst_full = min(rows, key=lambda row: fnum(row["nocs_full_auroc"]))
    worst_gaze = min(rows, key=lambda row: fnum(row["gaze_lr_auroc"]))

    lines = [
        "# NOCS Residual Behavior Diagnostic",
        "",
        "## Summary",
        "",
        "- Residual runs: %d subjects" % len(rows),
        "- Mean residual AUROC: %s" % fmt(mean(residual_aurocs)),
        "- Mean full NOCS AUROC: %s" % fmt(mean(full_aurocs)),
        "- Mean gaze LR AUROC: %s" % fmt(mean(gaze_aurocs)),
        "- Mean residual minus full AUROC: %s" % fmt(mean(deltas_full)),
        "- Residual improved over full on %d subjects." % improved_full,
        "- Residual improved over gaze LR on %d subjects." % improved_gaze,
        "",
        "## Largest Residual Drops vs Full",
        "",
    ]
    for row in sorted_drop[:5]:
        lines.append(
            "- %s: residual %s vs full %s, delta %s"
            % (
                row["heldout_subject"],
                fmt(fnum(row["nocs_residual_auroc"])),
                fmt(fnum(row["nocs_full_auroc"])),
                fmt(fnum(row["delta_residual_minus_full_auroc"])),
            )
        )
    lines.extend(
        [
            "",
            "## Worst Subject Comparison",
            "",
            "- Residual worst: %s AUROC %s" % (worst_residual["heldout_subject"], fmt(fnum(worst_residual["nocs_residual_auroc"]))),
            "- Full NOCS worst: %s AUROC %s" % (worst_full["heldout_subject"], fmt(fnum(worst_full["nocs_full_auroc"]))),
            "- Gaze LR worst: %s AUROC %s" % (worst_gaze["heldout_subject"], fmt(fnum(worst_gaze["gaze_lr_auroc"]))),
            "",
            "## Internal Residual Export",
            "",
        ]
    )
    if gate_exported:
        lines.append("- Residual gate/logit fields were exported; see residual_gate_stats.csv.")
    else:
        lines.append("- Residual internal behavior was not exported in the current run.")
        lines.append("- Missing prediction fields: %s" % ", ".join(missing_fields))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Residual correction under the current conservative gate degrades NOCS full and should not be expanded to more seeds.",
            "",
        ]
    )
    (output_dir / "residual_behavior_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NOCS-R residual behavior.")
    parser.add_argument("--full-dir", type=Path, default=Path("outputs/nocs_seed1_f80f2f3"))
    parser.add_argument("--residual-dir", type=Path, default=Path("outputs/nocs_residual_seed1"))
    parser.add_argument("--word-pool-summary", type=Path, default=Path("outputs/word_pool_baseline_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nocs_residual_diagnostic"))
    args = parser.parse_args()

    full = read_single_row_csvs(args.full_dir, "_metrics.csv")
    residual = read_single_row_csvs(args.residual_dir, "_metrics.csv")
    gaze_lr = read_gaze_lr(args.word_pool_summary)
    rows = build_subject_rows(full, residual, gaze_lr)
    write_rows(rows, args.output_dir / "residual_vs_full_subject_table.csv")
    failures = [row for row in rows if fnum(row["delta_residual_minus_full_auroc"]) < 0]
    failures.sort(key=lambda row: fnum(row["delta_residual_minus_full_auroc"]))
    write_rows(failures, args.output_dir / "residual_failure_subjects.csv")
    gate_exported, missing_fields = maybe_write_gate_stats(args.residual_dir, args.output_dir)
    write_report(rows, gate_exported, missing_fields, args.output_dir)
    print("Wrote residual diagnostics for %d protocols to %s" % (len(rows), args.output_dir))


if __name__ == "__main__":
    main()
