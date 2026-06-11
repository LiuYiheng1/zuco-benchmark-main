"""Run post-hoc CS-NOCS admission search without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adagtcn_aligned.nocs.cs_nocs import (
    apply_error_aware_strategy,
    apply_utility_strategy,
    choose_best_strategy,
    group_subject_rows,
    load_predictions,
    make_paper_ready_table,
    report_markdown,
    summarize_strategy,
    write_rows,
)


def run_search(args: argparse.Namespace) -> None:
    protocols = load_predictions(args.exact_dir, args.safe_dir)
    all_rows = []
    summaries = []

    epsilons = (0.0, 0.0025, 0.005, 0.01)
    for eps in epsilons:
        strategy = "utility_eps_%g" % eps
        rows = [
            dict(
                apply_utility_strategy(
                    item,
                    epsilon=eps,
                    require_reliability=False,
                    n_boot=args.n_boot,
                    seed=args.seed + idx,
                ),
                strategy=strategy,
            )
            for idx, item in enumerate(protocols)
        ]
        all_rows.extend(rows)
        summaries.append(summarize_strategy(strategy, rows))

    rows = [
        dict(
            apply_error_aware_strategy(item, n_boot=args.n_boot, seed=args.seed + idx),
            strategy="error_aware_admission",
        )
        for idx, item in enumerate(protocols)
    ]
    all_rows.extend(rows)
    summaries.append(summarize_strategy("error_aware_admission", rows))

    for eps in epsilons:
        strategy = "reliability_utility_eps_%g" % eps
        rows = [
            dict(
                apply_utility_strategy(
                    item,
                    epsilon=eps,
                    require_reliability=True,
                    n_boot=args.n_boot,
                    seed=args.seed + idx,
                ),
                strategy=strategy,
            )
            for idx, item in enumerate(protocols)
        ]
        all_rows.extend(rows)
        summaries.append(summarize_strategy(strategy, rows))

    best = choose_best_strategy(summaries)
    grouped = group_subject_rows(all_rows)
    best_rows = grouped[str(best["strategy"])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(summaries, args.output_dir / "cs_nocs_strategy_summary.csv")
    write_rows(all_rows, args.output_dir / "cs_nocs_subject_table.csv")
    write_rows(best_rows, args.output_dir / "cs_nocs_best_strategy_subjects.csv")
    write_rows(
        make_paper_ready_table(best, args.exact_dir, args.safe_dir, args.nocs_dir),
        args.output_dir / "paper_ready_main_table.csv",
    )
    (args.output_dir / "CS_NOCS_AUTONOMOUS_REPORT.md").write_text(
        report_markdown(best, best_rows, summaries),
        encoding="utf-8",
    )
    print("Read %d protocols" % len(protocols), flush=True)
    print("Best strategy: %s" % best["strategy"], flush=True)
    print("Mean AUROC: %.6f" % best["mean_auroc"], flush=True)
    print("Negative transfer subjects: %d" % best["negative_transfer_subject_count"], flush=True)
    print("Wrote %s" % args.output_dir, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc CS-NOCS admission search.")
    parser.add_argument("--exact_dir", type=Path, required=True)
    parser.add_argument("--safe_dir", type=Path, required=True)
    parser.add_argument("--nocs_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cs_nocs_search"))
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
