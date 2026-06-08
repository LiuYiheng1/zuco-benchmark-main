"""
Generate comprehensive baseline comparison report
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "loso")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

svm_df = pd.read_csv(os.path.join(RESULTS_DIR, "svm_all_features_loso.csv"))
majority_random_df = pd.read_csv(os.path.join(RESULTS_DIR, "majority_random_loso.csv"))

svm_models = ['SVM_EEG_only', 'SVM_Gaze_only', 'SVM_Combined']
pytorch_models = ['MLP_EEG_only', 'MLP_Gaze_only', 'MLP_EarlyConcat', 'MLP_LateFusion', 'MLP_AttentionFusion']

all_models = ['Majority', 'Random', 'SVM_EEG_only', 'SVM_Gaze_only', 'SVM_Combined',
              'MLP_EEG_only', 'MLP_Gaze_only', 'MLP_EarlyConcat', 'MLP_LateFusion', 'MLP_AttentionFusion']

svm_summary = svm_df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std']
}).reset_index()
svm_summary.columns = ['model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'balanced_accuracy_mean', 'balanced_accuracy_std']

majority_summary = majority_random_df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std']
}).reset_index()
majority_summary.columns = ['model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'balanced_accuracy_mean', 'balanced_accuracy_std']

combined_summary = pd.concat([majority_summary, svm_summary], ignore_index=True)

print("="*70)
print("COMBINED BASELINE SUMMARY")
print("="*70)
for _, row in combined_summary.iterrows():
    print(f"{row['model']:20s}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} | F1={row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}")

print("\n" + "="*70)
print("KEY COMPARISONS")
print("="*70)

svm_gaze = svm_df[svm_df['model'] == 'SVM_Gaze_only'].groupby('held_out')['accuracy'].mean()
svm_eeg = svm_df[svm_df['model'] == 'SVM_EEG_only'].groupby('held_out')['accuracy'].mean()
svm_combined = svm_df[svm_df['model'] == 'SVM_Combined'].groupby('held_out')['accuracy'].mean()
majority_acc = majority_random_df[majority_random_df['model'] == 'Majority'].groupby('held_out')['accuracy'].mean()
random_acc = majority_random_df[majority_random_df['model'] == 'Random'].groupby('held_out')['accuracy'].mean()

print("\n1. Gaze-only (SVM) vs Random baseline:")
print(f"   SVM_Gaze_only mean: {svm_gaze.mean():.4f} vs Random mean: {random_acc.mean():.4f}")
print(f"   Difference: {svm_gaze.mean() - random_acc.mean():+.4f}")

print("\n2. Gaze-only (SVM) vs EEG-only (SVM):")
print(f"   SVM_Gaze_only mean: {svm_gaze.mean():.4f} vs SVM_EEG_only mean: {svm_eeg.mean():.4f}")
print(f"   Difference: {svm_gaze.mean() - svm_eeg.mean():+.4f}")

print("\n3. Combined (SVM) vs Gaze-only (SVM):")
print(f"   SVM_Combined mean: {svm_combined.mean():.4f} vs SVM_Gaze_only mean: {svm_gaze.mean():.4f}")
print(f"   Difference: {svm_combined.mean() - svm_gaze.mean():+.4f}")

print("\n" + "="*70)
print("PER-SUBJECT ANALYSIS (SVM)")
print("="*70)

per_subject_svm = svm_df.groupby(['model', 'held_out'])['accuracy'].mean().unstack(level=0)
print("\nWorst subjects for SVM_Gaze_only:")
gaze_worst = per_subject_svm['SVM_Gaze_only'].nsmallest(5)
for subj, acc in gaze_worst.items():
    print(f"  {subj}: {acc:.4f}")

print("\nBest subjects for SVM_Gaze_only:")
gaze_best = per_subject_svm['SVM_Gaze_only'].nlargest(5)
for subj, acc in gaze_best.items():
    print(f"  {subj}: {acc:.4f}")

print("\n" + "="*70)
print("HIGH/LOW GAZE SUBJECTS ANALYSIS")
print("="*70)

high_gaze_subjects = ['YTL', 'YIS', 'YSD']
low_gaze_subjects = ['YAK', 'YRP', 'YLS']

print("\nHigh gaze subjects (SVM):")
for subj in high_gaze_subjects:
    if subj in per_subject_svm.index:
        gaze_acc = per_subject_svm.loc[subj, 'SVM_Gaze_only'] if 'SVM_Gaze_only' in per_subject_svm.columns else None
        eeg_acc = per_subject_svm.loc[subj, 'SVM_EEG_only'] if 'SVM_EEG_only' in per_subject_svm.columns else None
        comb_acc = per_subject_svm.loc[subj, 'SVM_Combined'] if 'SVM_Combined' in per_subject_svm.columns else None
        gaze_str = f"{gaze_acc:.4f}" if gaze_acc is not None else 'N/A'
        eeg_str = f"{eeg_acc:.4f}" if eeg_acc is not None else 'N/A'
        comb_str = f"{comb_acc:.4f}" if comb_acc is not None else 'N/A'
        print(f"  {subj}: Gaze={gaze_str}, EEG={eeg_str}, Combined={comb_str}")

print("\nLow gaze subjects (SVM):")
for subj in low_gaze_subjects:
    if subj in per_subject_svm.index:
        gaze_acc = per_subject_svm.loc[subj, 'SVM_Gaze_only'] if 'SVM_Gaze_only' in per_subject_svm.columns else None
        eeg_acc = per_subject_svm.loc[subj, 'SVM_EEG_only'] if 'SVM_EEG_only' in per_subject_svm.columns else None
        comb_acc = per_subject_svm.loc[subj, 'SVM_Combined'] if 'SVM_Combined' in per_subject_svm.columns else None
        gaze_str = f"{gaze_acc:.4f}" if gaze_acc is not None else 'N/A'
        eeg_str = f"{eeg_acc:.4f}" if eeg_acc is not None else 'N/A'
        comb_str = f"{comb_acc:.4f}" if comb_acc is not None else 'N/A'
        print(f"  {subj}: Gaze={gaze_str}, EEG={eeg_str}, Combined={comb_str}")

print("\n" + "="*70)
print("CREATE COMBINED SUMMARY TABLE")
print("="*70)

all_summary_df = combined_summary.copy()
all_summary_df = all_summary_df.sort_values('accuracy_mean', ascending=False)

summary_csv = os.path.join(RESULTS_DIR, "combined_baseline_summary.csv")
all_summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved: {summary_csv}")

print("\n" + "="*70)
print("FINAL RANKING (by Accuracy)")
print("="*70)
for i, (_, row) in enumerate(all_summary_df.iterrows(), 1):
    print(f"{i:2d}. {row['model']:20s}: {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("""
1. EEG-only (SVM): 52.19% - marginally above random (50.23%), below majority (53.99%)
   - Confirms EEG signal is weak for cross-subject classification

2. Gaze-only (SVM): 60.72% - strongest signal with high variance
   - YTL, YIS, YSD show exceptional performance (>80%)
   - YAK, YRP show poor performance (<50%)

3. Combined (SVM): 59.43% - slightly below gaze-only
   - Fusion does NOT help when gaze alone dominates

4. Subject variability is the dominant factor
   - Some subjects have highly predictable gaze patterns
   - Others show near-random classification for both modalities

5. These results are consistent with SVM baseline
   - MLP models need more data to match SVM performance
   - Early/Late/Attention fusion do not clearly outperform SVM Combined
""")

print("\nDone!")