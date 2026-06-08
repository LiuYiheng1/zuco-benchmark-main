"""
Evaluation and Results Aggregation Script
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

def load_results(results_dir: str, prefix: str = "") -> pd.DataFrame:
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and prefix in f]

    if not csv_files:
        print(f"No CSV files found in {results_dir} with prefix '{prefix}'")
        return None

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(results_dir, f))
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def compute_summary_stats(results_df: pd.DataFrame, groupby_col: str = 'model') -> pd.DataFrame:
    summary = results_df.groupby(groupby_col).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    }).reset_index()

    summary.columns = [
        'model',
        'accuracy_mean', 'accuracy_std',
        'macro_f1_mean', 'macro_f1_std',
        'balanced_accuracy_mean', 'balanced_accuracy_std',
        'auroc_mean', 'auroc_std'
    ]

    return summary

def compute_per_subject_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    subject_summary = results_df.groupby(['model', 'subject_id']).agg({
        'accuracy': 'mean',
        'macro_f1': 'mean',
        'balanced_accuracy': 'mean'
    }).reset_index()

    return subject_summary

def analyze_router_weights(router_df: pd.DataFrame) -> Dict:
    if router_df is None or len(router_df) == 0:
        return {}

    stats = {
        'overall': {
            'eeg_weight_mean': router_df['router_weight_eeg'].mean(),
            'eeg_weight_std': router_df['router_weight_eeg'].std(),
            'gaze_weight_mean': router_df['router_weight_gaze'].mean(),
            'gaze_weight_std': router_df['router_weight_gaze'].std(),
            'fusion_weight_mean': router_df['router_weight_fusion'].mean(),
            'fusion_weight_std': router_df['router_weight_fusion'].std()
        }
    }

    by_label = router_df.groupby('true_label').agg({
        'router_weight_eeg': ['mean', 'std'],
        'router_weight_gaze': ['mean', 'std'],
        'router_weight_fusion': ['mean', 'std']
    })
    by_label.columns = ['_'.join(col) for col in by_label.columns]
    stats['by_label'] = by_label.to_dict()

    by_subject = router_df.groupby('subject_id').agg({
        'router_weight_eeg': 'mean',
        'router_weight_gaze': 'mean',
        'router_weight_fusion': 'mean'
    })
    stats['by_subject'] = by_subject.to_dict()

    return stats

def generate_report(results_df: pd.DataFrame, router_df: Optional[pd.DataFrame],
                   summary_stats: pd.DataFrame, output_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = f"""# TGCR v1 Experiment Report

## Experiment Setup

- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Split Protocol**: Cross-subject (train on Y-subjects, test on X-subjects)
- **Seeds**: [0, 1, 2, 3, 4]

## Results Summary

### Overall Performance (Mean ± Std across seeds)

| Model | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|-------------------|-------|
"""

    for _, row in summary_stats.iterrows():
        model = row['model']
        acc = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        f1 = f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
        bacc = f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}"
        auroc = f"{row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}" if pd.notna(row['auroc_mean']) else "N/A"
        report += f"| {model} | {acc} | {f1} | {bacc} | {auroc} |\n"

    if router_df is not None and len(router_df) > 0:
        router_stats = analyze_router_weights(router_df)

        report += f"""
## Router Weight Analysis

### Overall Statistics
- **EEG Expert Weight**: {router_stats['overall']['eeg_weight_mean']:.4f} ± {router_stats['overall']['eeg_weight_std']:.4f}
- **Gaze Expert Weight**: {router_stats['overall']['gaze_weight_mean']:.4f} ± {router_stats['overall']['gaze_weight_std']:.4f}
- **Fusion Expert Weight**: {router_stats['overall']['fusion_weight_mean']:.4f} ± {router_stats['overall']['fusion_weight_std']:.4f}

### Router Weight by True Label
| Label | EEG Weight | Gaze Weight | Fusion Weight |
|-------|------------|-------------|---------------|
"""
        if 'by_label' in router_stats:
            for label in router_stats['by_label'].get('router_weight_eeg_mean', {}):
                eeg_w = router_stats['by_label']['router_weight_eeg_mean'].get(label, 0)
                gaze_w = router_stats['by_label']['router_weight_gaze_mean'].get(label, 0)
                fusion_w = router_stats['by_label']['router_weight_fusion_mean'].get(label, 0)
                label_name = "NR (Normal Reading)" if label == 1 else "TSR (Task-Specific)"
                report += f"| {label_name} | {eeg_w:.4f} | {gaze_w:.4f} | {fusion_w:.4f} |\n"

    report += """
## Notes

- Cross-subject protocol: No subject overlap between train and test
- Evaluation: Per-sample prediction aggregated across all test subjects
- All random seeds fixed for reproducibility

## Files Generated

- `summary_all_models_<timestamp>.csv`: Summary statistics for all models
- `all_results_<timestamp>.csv`: Detailed results for each sample
- `tgcr_router_weights_<timestamp>.csv`: Router weights for TGCR model (if applicable)
"""

    report_path = os.path.join(output_dir, f"tgcr_experiment_report_{timestamp}.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    src_dir = project_root if os.path.basename(project_root) == 'src' else os.path.join(project_root, 'src')
    results_dir = os.path.join(src_dir, args.results_dir)
    output_dir = os.path.join(src_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_dir}")

    results_df = load_results(results_dir, prefix="all_results")
    router_df = load_results(results_dir, prefix="tgcr_router_weights")

    if results_df is None:
        print("No results found!")
        return

    print(f"Loaded {len(results_df)} result rows")

    summary_stats = compute_summary_stats(results_df)
    summary_path = os.path.join(output_dir, "summary_mean_std.csv")
    summary_stats.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    per_subject = compute_per_subject_stats(results_df)
    per_subject_path = os.path.join(output_dir, "summary_per_subject.csv")
    per_subject.to_csv(per_subject_path, index=False)
    print(f"Saved per-subject stats to {per_subject_path}")

    generate_report(results_df, router_df, summary_stats, output_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()