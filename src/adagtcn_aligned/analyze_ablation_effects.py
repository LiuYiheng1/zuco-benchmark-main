"""Audit whether CNO-GSM modules produce real measured gains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ORDER = [
    "eeg_only_graph_tcn",
    "eeg_graph_ssm",
    "gaze_only_ssm",
    "adagtcn_aligned",
    "gaze_control_ssm",
    "bipartite_graph_ssm",
    "bridge_bipartite_ssm",
    "full_cnogsm",
]

MODULE_LABEL = {
    "eeg_only_graph_tcn": "EEG graph temporal baseline",
    "eeg_graph_ssm": "Isolation: EEG graph with state-space temporal encoder",
    "gaze_only_ssm": "Isolation: gaze-only state-space encoder",
    "adagtcn_aligned": "AdaGTCN-aligned EEG+gaze temporal baseline",
    "gaze_control_ssm": "Module 1: gaze-controlled state-space temporal encoder",
    "bipartite_graph_ssm": "Module 2: neuro-oculomotor bipartite graph",
    "bridge_bipartite_ssm": "Module 3: subject-invariant bridge",
    "full_cnogsm": "Modules 1-5: full CNO-GSM",
}


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source"] = str(path)
                rows.append(row)
    return rows


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except Exception:
        return 0.0


def group_by_model(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        model = row["model"]
        grouped.setdefault(model, []).append(as_float(row, metric))
    return grouped


def audit(rows: list[dict[str, Any]], metric: str, min_delta: float) -> list[dict[str, Any]]:
    grouped = group_by_model(rows, metric)
    baseline = mean(grouped.get("adagtcn_aligned", [0.0]))
    results = []
    previous = None
    for model in ORDER:
        if model not in grouped:
            continue
        values = grouped[model]
        score = mean(values)
        delta_vs_baseline = score - baseline
        delta_vs_previous = None if previous is None else score - previous
        effective = delta_vs_baseline >= min_delta if model != "adagtcn_aligned" else None
        results.append(
            {
                "model": model,
                "module": MODULE_LABEL[model],
                "metric": metric,
                "mean": score,
                "n_runs": len(values),
                "delta_vs_adagtcn_aligned": delta_vs_baseline,
                "delta_vs_previous": delta_vs_previous,
                "effective_vs_adagtcn_aligned": effective,
            }
        )
        previous = score
    return results


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# CNO-GSM Ablation Effect Audit\n",
        "\n",
        "A module is marked effective only when its measured mean score exceeds the "
        "AdaGTCN-aligned baseline by the configured threshold. This file is an "
        "audit aid, not a substitute for full multi-seed statistical testing.\n",
        "\n",
        "| Model | Module | Mean | Delta vs AdaGTCN-aligned | Effective |\n",
        "|---|---|---:|---:|---|\n",
    ]
    for row in rows:
        effective = row["effective_vs_adagtcn_aligned"]
        effective_text = "baseline" if effective is None else ("yes" if effective else "no")
        lines.append(
            "| %s | %s | %.4f | %.4f | %s |\n"
            % (row["model"], row["module"], row["mean"], row["delta_vs_adagtcn_aligned"], effective_text)
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CNO-GSM ablation effects.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--metric", default="test_macro_f1")
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/adagtcn_aligned"))
    args = parser.parse_args()

    rows = audit(read_rows(args.inputs), args.metric, args.min_delta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cnogsm_ablation_effect_audit.csv"
    md_path = args.output_dir / "cnogsm_ablation_effect_audit.md"
    json_path = args.output_dir / "cnogsm_ablation_effect_audit.json"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote %s" % csv_path)
    print("Wrote %s" % md_path)
    print("Wrote %s" % json_path)


if __name__ == "__main__":
    main()
