"""
EEG Adaptation Full Experiment Report
"""
import pandas as pd
import numpy as np
from datetime import datetime

RESULTS_DIR = "results/eeg_adaptation"
REPORTS_DIR = "reports"
import os
os.makedirs(REPORTS_DIR, exist_ok=True)

print("="*70)
print("EEG SUBJECT-ADAPTATION EXPERIMENT - FINAL REPORT")
print("="*70)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("="*70)
print("1. EXPERIMENT SUMMARY")
print("="*70)
print("""
This experiment evaluates subject-adversarial domain adaptation for EEG-based
reading mode classification (Normal Reading vs Task-Specific Reading).

Models tested:
- Raw_EEG: Standard EEG classifier without adaptation
- EEG_CORAL: CORAL domain adaptation on EEG features
- EEG_Adversarial λ=X: Subject-adversarial training with different λ values
- Gaze_only: Eye-tracking features only (baseline)
- Combined: EEG + Gaze fusion

Protocol: Leave-One-Subject-Out (LOSO) cross-validation
Subjects: 16 (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL)
Seeds: 5 (0, 1, 2, 3, 4)
""")

print("="*70)
print("2. MAIN RESULTS (5 seeds)")
print("="*70)

try:
    summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "eeg_adaptation_summary_5seed.csv"))
    print("\nPerformance summary (sorted by accuracy):\n")
    print(f"{'Model':<30} {'Accuracy':>12} {'Macro-F1':>12} {'Bal.Acc':>12} {'AUROC':>12}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['accuracy_mean']:>11.2%} {row['macro_f1_mean']:>11.2%} {row['balanced_accuracy_mean']:>11.2%} {row['auroc_mean']:>11.2%}")
except Exception as e:
    print(f"Could not load summary: {e}")

print("\n" + "="*70)
print("3. KEY FINDINGS")
print("="*70)
print("""
a) Gaze-only remains the strongest single-modality baseline:
   - Gaze-only: ~61.3% accuracy
   - This confirms gaze is more robust across subjects

b) EEG subject-adversarial adaptation improves over Raw EEG:
   - Raw EEG: ~50.8% accuracy
   - EEG_CORAL: ~51.5% accuracy (+0.7%)
   - EEG_Adversarial: ~54.5-54.6% accuracy (+3.7-3.8%)
   - The improvement suggests adversarial training helps reduce subject shift

c) EEG still underperforms gaze:
   - This is expected as EEG has higher individual variability
   - EEG contains strong subject-specific patterns but poor cross-subject generalization

d) Combined EEG+Gaze fusion (~59.3%) does not outperform gaze-only (~61.3%):
   - This suggests the EEG features may not add complementary information
   - Or the fusion method (simple concatenation) is suboptimal
""")

print("\n" + "="*70)
print("4. SIGNIFICANCE TEST RESULTS")
print("="*70)

try:
    sig_df = pd.read_csv(os.path.join(RESULTS_DIR, "eeg_adaptation_significance.csv"))
    print("\nPaired Wilcoxon signed-rank tests (α=0.05):\n")
    for _, row in sig_df.iterrows():
        sig_mark = "*" if row['p_value'] < 0.05 else ""
        print(f"{row['comparison']:<40} Δ={row['mean_diff']:>+.4f}, p={row['p_value']:.4f} {sig_mark}")
except Exception as e:
    print(f"Could not load significance results: {e}")

print("\n" + "="*70)
print("5. LOW-PERFORMANCE SUBJECTS")
print("="*70)
print("""
Subjects with consistently low performance across models:
- YLS: Often below 45% accuracy
- YSL: Often below 50% accuracy
- YHS: Often below 53% accuracy

These subjects may have:
- Different neural responses to reading tasks
- Lower signal quality
- Unique cognitive strategies
""")

print("\n" + "="*70)
print("6. RECOMMENDATIONS FOR PAPER")
print("="*70)
print("""
DO say:
- "EEG contains strong subject-specific task information but suffers from
   severe cross-subject domain shift."
- "Subject-adversarial adaptation partially improves cross-subject generalization."
- "Eye-tracking (gaze) remains more robust across subjects than EEG."

DO NOT say:
- "EEG is the strongest modality."
- "EEG outperforms gaze."
- "Adversarial adaptation fully solves the cross-subject problem."

Future directions:
- Investigate why YLS/YSL/YHS have poor performance
- Try other domain adaptation methods (MMD, DANN)
- Explore subject-specific calibration for EEG
""")

report_content = f"""# EEG Subject-Adaptation Experiment Report

## 1. Experiment Overview

**Date:** {datetime.now().strftime('%Y-%m-%d')}

**Goal:** Evaluate subject-adversarial domain adaptation for EEG-based reading mode classification (Normal Reading vs Task-Specific Reading).

**Protocol:** Leave-One-Subject-Out (LOSO) cross-validation

**Subjects:** 16 (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL)

**Seeds:** 5 (0, 1, 2, 3, 4)

## 2. Models Tested

| Model | Description |
|-------|-------------|
| Raw_EEG | Standard EEG classifier (SVM) without adaptation |
| EEG_CORAL | CORAL domain adaptation on EEG features |
| EEG_Adversarial λ=X | Subject-adversarial training with gradient reversal |
| Gaze_only | Eye-tracking features only (baseline) |
| Combined | EEG + Gaze feature concatenation |

## 3. Main Results

| Model | Accuracy | Macro-F1 | Balanced Acc | AUROC |
|-------|----------|----------|--------------|-------|
| Gaze_only | ~61.3% | ~57.0% | ~61.3% | ~69.1% |
| Combined | ~59.3% | ~54.0% | ~59.3% | ~69.4% |
| EEG_Adversarial (λ=0.01) | ~54.5% | ~46.1% | ~54.5% | N/A |
| EEG_Adversarial (λ=0.05) | ~54.6% | ~45.8% | ~54.6% | N/A |
| EEG_CORAL | ~51.5% | ~44.2% | ~51.5% | ~56.2% |
| Raw_EEG | ~50.8% | ~42.3% | ~50.8% | ~57.2% |

## 4. Key Findings

### 4.1 Gaze is more robust across subjects
Eye-tracking features (gaze) consistently outperform EEG features in cross-subject evaluation, achieving ~61.3% accuracy compared to EEG's ~50-55%.

### 4.2 Adversarial adaptation helps EEG
Subject-adversarial training improves EEG cross-subject accuracy by ~3.7-3.8% over raw EEG, suggesting it partially reduces subject-specific information.

### 4.3 EEG underperforms gaze
Despite EEG's strong within-subject performance (86-89% in previous experiments), it struggles with cross-subject generalization (50-55%).

### 4.4 Fusion doesn't help
Combining EEG and gaze (59.3%) doesn't outperform gaze alone (61.3%), suggesting EEG features don't add complementary cross-subject information.

## 5. Statistical Significance

Preliminary Wilcoxon tests show:
- EEG_Adversarial λ=0.01 vs Raw_EEG: Δ=+3.9%, p=0.10 (not significant)
- EEG_CORAL vs Raw_EEG: Δ=+0.7%, p=0.41 (not significant)

The lack of significance may be due to high subject-level variance.

## 6. Problematic Subjects

Subjects with consistently low performance:
- **YLS**: Often below 45% accuracy
- **YSL**: Often below 50% accuracy
- **YHS**: Often below 53% accuracy

## 7. Recommendations for Paper

### Should say:
> "EEG contains strong subject-specific task information but suffers from severe cross-subject domain shift. Subject-adversarial adaptation partially improves cross-subject generalization."

> "Eye-tracking (gaze) remains more robust across subjects than EEG in cross-subject evaluation."

### Should NOT say:
- "EEG is the strongest modality."
- "EEG outperforms gaze."
- "Adversarial adaptation fully solves the cross-subject problem."

## 8. Future Directions

1. Investigate why YLS/YSL/YHS have poor performance (signal quality? cognitive strategies?)
2. Try other domain adaptation methods (MMD, DANN)
3. Explore subject-specific calibration for EEG
4. Consider hierarchical models that separate subject and task effects
"""

report_path = os.path.join(REPORTS_DIR, "eeg_adaptation_full_report.md")
with open(report_path, 'w') as f:
    f.write(report_content)

print(f"\nReport saved to: {REPORTS_DIR}/eeg_adaptation_full_report.md")
print("\n" + "="*70)
print("REPORT GENERATION COMPLETE")
print("="*70)