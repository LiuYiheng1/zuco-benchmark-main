"""Summarize NOCS experiment outputs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


BASELINES = {
    "gaze_word_pool_lr_auroc": 0.6939,
    "gaze_word_pool_lr_macro_f1": 0.5739,
    "eeg_word_pool_lr_auroc": 0.5243,
    "concat_word_pool_lr_auroc": 0.5369,
    "nocs_full_auroc": 0.6922,
    "nocs_full_macro_f1": 0.5458,
    "nocs_gaze_only_auroc": 0.5892,
    "nocs_gaze_only_macro_f1": 0.5134,
    "nocs_residual_auroc": 0.6300,
    "nocs_residual_macro_f1": 0.4967,
}


def read_metrics(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*_metrics.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out = []
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "stat_auroc",
        "stat_macro_f1",
        "full_minus_stat_auroc",
        "full_minus_stat_macro_f1",
        "mean_residual_gate",
        "mean_residual_norm",
        "anchor_test_auroc",
        "anchor_test_macro_f1",
        "corrected_test_auroc",
        "corrected_test_macro_f1",
        "safe_test_auroc",
        "safe_test_macro_f1",
        "safe_minus_anchor_test_auroc",
        "safe_minus_corrected_test_auroc",
        "selected_alpha",
    ]
    for group, items in sorted(grouped.items()):
        row: dict[str, Any] = {key: group, "n_runs": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) not in {None, ""}]
            if values:
                row[metric + "_mean"] = mean(values)
                row[metric + "_std"] = stdev(values) if len(values) > 1 else 0.0
                row[metric + "_min"] = min(values)
        out.append(row)
    return out


def bootstrap_ci(values: list[float], baseline: float, n_boot: int = 2000) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    rng = np.random.RandomState(0)
    diffs = []
    arr = np.asarray(values, dtype=np.float64) - baseline
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        diffs.append(float(sample.mean()))
    return float(arr.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def wilcoxon_p(values: list[float], baseline: float) -> float:
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(np.asarray(values) - baseline).pvalue)
    except Exception:
        return math.nan


def paired_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    ablations = sorted({str(row.get("ablation", "")) for row in rows if row.get("ablation")})
    comparisons = [
        ("gaze_word_pool_lr", BASELINES["gaze_word_pool_lr_auroc"], BASELINES["gaze_word_pool_lr_macro_f1"]),
        ("eeg_word_pool_lr", BASELINES["eeg_word_pool_lr_auroc"], math.nan),
        ("concat_word_pool_lr", BASELINES["concat_word_pool_lr_auroc"], math.nan),
        ("nocs_full", BASELINES["nocs_full_auroc"], BASELINES["nocs_full_macro_f1"]),
        ("nocs_gaze_only", BASELINES["nocs_gaze_only_auroc"], BASELINES["nocs_gaze_only_macro_f1"]),
        ("nocs_residual", BASELINES["nocs_residual_auroc"], BASELINES["nocs_residual_macro_f1"]),
    ]
    for ablation in ablations:
        ablation_rows = [row for row in rows if row.get("ablation") == ablation]
        aurocs = [float(row["test_auroc"]) for row in ablation_rows]
        macro_f1s = [float(row["test_macro_f1"]) for row in ablation_rows]
        for name, auroc_base, f1_base in comparisons:
            diff, lo, hi = bootstrap_ci(aurocs, auroc_base)
            row: dict[str, Any] = {
                "comparison": "%s_vs_%s" % (ablation, name),
                "ablation": ablation,
                "n": len(ablation_rows),
                "auroc_baseline": auroc_base,
                "auroc_diff_mean": diff,
                "auroc_diff_ci_low": lo,
                "auroc_diff_ci_high": hi,
                "auroc_wilcoxon_p": wilcoxon_p(aurocs, auroc_base),
                "subjects_above_baseline": sum(1 for value in aurocs if value > auroc_base),
                "worst_subject_auroc": min(aurocs) if aurocs else math.nan,
                "std_across_subjects_auroc": stdev(aurocs) if len(aurocs) > 1 else 0.0,
            }
            if not math.isnan(f1_base):
                f1_diff, f1_lo, f1_hi = bootstrap_ci(macro_f1s, f1_base)
                row.update(
                    {
                        "macro_f1_baseline": f1_base,
                        "macro_f1_diff_mean": f1_diff,
                        "macro_f1_diff_ci_low": f1_lo,
                        "macro_f1_diff_ci_high": f1_hi,
                        "macro_f1_wilcoxon_p": wilcoxon_p(macro_f1s, f1_base),
                    }
                )
            out.append(row)
        stat_aurocs = [float(row["stat_auroc"]) for row in ablation_rows if row.get("stat_auroc") not in {None, ""}]
        stat_macro_f1s = [float(row["stat_macro_f1"]) for row in ablation_rows if row.get("stat_macro_f1") not in {None, ""}]
        if len(stat_aurocs) == len(ablation_rows) and stat_aurocs:
            diffs = [float(row["test_auroc"]) - float(row["stat_auroc"]) for row in ablation_rows]
            f1_diffs = [float(row["test_macro_f1"]) - float(row["stat_macro_f1"]) for row in ablation_rows]
            out.append(
                {
                    "comparison": "%s_full_logits_vs_stat_branch" % ablation,
                    "ablation": ablation,
                    "n": len(ablation_rows),
                    "auroc_diff_mean": mean(diffs),
                    "auroc_diff_ci_low": float(np.percentile(diffs, 2.5)),
                    "auroc_diff_ci_high": float(np.percentile(diffs, 97.5)),
                    "subjects_above_baseline": sum(1 for value in diffs if value > 0),
                    "macro_f1_diff_mean": mean(f1_diffs),
                }
            )
            diff, lo, hi = bootstrap_ci(stat_aurocs, BASELINES["gaze_word_pool_lr_auroc"])
            f1_diff, f1_lo, f1_hi = bootstrap_ci(stat_macro_f1s, BASELINES["gaze_word_pool_lr_macro_f1"])
            out.append(
                {
                    "comparison": "%s_stat_branch_vs_gaze_word_pool_lr" % ablation,
                    "ablation": ablation,
                    "n": len(ablation_rows),
                    "auroc_baseline": BASELINES["gaze_word_pool_lr_auroc"],
                    "auroc_diff_mean": diff,
                    "auroc_diff_ci_low": lo,
                    "auroc_diff_ci_high": hi,
                    "subjects_above_baseline": sum(1 for value in stat_aurocs if value > BASELINES["gaze_word_pool_lr_auroc"]),
                    "macro_f1_baseline": BASELINES["gaze_word_pool_lr_macro_f1"],
                    "macro_f1_diff_mean": f1_diff,
                    "macro_f1_diff_ci_low": f1_lo,
                    "macro_f1_diff_ci_high": f1_hi,
                    "macro_f1_wilcoxon_p": wilcoxon_p(stat_macro_f1s, BASELINES["gaze_word_pool_lr_macro_f1"]),
                }
            )
        safe_minus_anchor = [
            float(row["safe_minus_anchor_test_auroc"])
            for row in ablation_rows
            if row.get("safe_minus_anchor_test_auroc") not in {None, ""}
        ]
        if len(safe_minus_anchor) == len(ablation_rows) and safe_minus_anchor:
            out.append(
                {
                    "comparison": "%s_safe_vs_anchor" % ablation,
                    "ablation": ablation,
                    "n": len(ablation_rows),
                    "auroc_diff_mean": mean(safe_minus_anchor),
                    "auroc_diff_ci_low": float(np.percentile(safe_minus_anchor, 2.5)),
                    "auroc_diff_ci_high": float(np.percentile(safe_minus_anchor, 97.5)),
                    "subjects_above_baseline": sum(1 for value in safe_minus_anchor if value > 0),
                    "subjects_equal_baseline": sum(1 for value in safe_minus_anchor if value == 0),
                    "subjects_below_baseline": sum(1 for value in safe_minus_anchor if value < 0),
                }
            )
    return out


def subject_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "heldout_subject": row.get("heldout_subject", ""),
                "protocol": row.get("protocol", ""),
                "seed": row.get("seed", ""),
                "ablation": row.get("ablation", ""),
                "test_auroc": row.get("test_auroc", ""),
                "test_macro_f1": row.get("test_macro_f1", ""),
                "test_accuracy": row.get("test_accuracy", ""),
                "anchor_test_auroc": row.get("anchor_test_auroc", ""),
                "corrected_test_auroc": row.get("corrected_test_auroc", ""),
                "safe_test_auroc": row.get("safe_test_auroc", ""),
                "safe_minus_anchor_test_auroc": row.get("safe_minus_anchor_test_auroc", ""),
                "selected_alpha": row.get("selected_alpha", ""),
                "selected_mode": row.get("selected_mode", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NOCS metrics.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/nocs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    rows = read_metrics(args.input_dir)
    write_rows(rows, args.output_dir / "nocs_summary.csv")
    write_rows(aggregate(rows, "ablation"), args.output_dir / "nocs_summary_by_ablation.csv")
    write_rows(subject_table(rows), args.output_dir / "nocs_subject_table.csv")
    write_rows(paired_stats(rows), args.output_dir / "nocs_paired_stats_vs_baselines.csv")
    print("Read %d NOCS metric rows" % len(rows))


if __name__ == "__main__":
    main()
