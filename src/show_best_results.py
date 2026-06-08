"""
Complete results from the best CAGF version: eeg_gaze_pilot_results.csv
PCET + GETA + CAGF
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

RESULTS_DIR = "results/final"

df = pd.read_csv(os.path.join(RESULTS_DIR, 'eeg_gaze_pilot_results.csv'))

k_values = [3, 5, 10, 20, 50]

print("="*100)
print("BEST CAGF VERSION RESULTS: PCET + GETA + CAGF")
print("="*100)

print("\n### 完整结果表格 (Accuracy %)")
print("-" * 100)

methods_order = [
    'EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP',
    'EEG+Gaze_concat', 'Static_EEG_Gaze_avg',
    'PCET_only', 'GETA_only',
    'PCET+GETA_concat', 'PCET+GETA_static_avg',
    'PCET+GETA+CAGF'
]

print(f"{'Method':<30} | {'k=3':>10} | {'k=5':>10} | {'k=10':>10} | {'k=20':>10} | {'k=50':>10}")
print("-" * 100)

for method in methods_order:
    acc_col = f'{method}_acc'
    if acc_col not in df.columns:
        continue
    vals = []
    for k in k_values:
        subset = df[df['n_cal'] == k]
        if len(subset) > 0:
            mean_val = subset[acc_col].mean() * 100
            std_val = subset[acc_col].std() * 100
            vals.append(f"{mean_val:.1f}±{std_val:.1f}")
        else:
            vals.append("N/A")
    print(f"{method:<30} | {' | '.join(vals)}")

print("\n" + "="*100)
print("PCET+GETA+CAGF 详细指标")
print("="*100)

print(f"\n{'k':>5} | {'Accuracy':>12} | {'Macro-F1':>12} | {'BAcc':>12} | {'AUROC':>12}")
print("-" * 60)

for k in k_values:
    subset = df[df['n_cal'] == k]
    if len(subset) > 0:
        acc = subset['PCET+GETA+CAGF_acc'].mean() * 100
        acc_std = subset['PCET+GETA+CAGF_acc'].std() * 100
        f1 = subset['PCET+GETA+CAGF_f1'].mean() * 100
        f1_std = subset['PCET+GETA+CAGF_f1'].std() * 100
        bacc = subset['PCET+GETA+CAGF_bacc'].mean() * 100
        bacc_std = subset['PCET+GETA+CAGF_bacc'].std() * 100
        auroc = subset['PCET+GETA+CAGF_auroc'].mean() * 100
        auroc_std = subset['PCET+GETA+CAGF_auroc'].std() * 100
        print(f"{k:>5} | {acc:>6.2f}±{acc_std:>4.1f} | {f1:>6.2f}±{f1_std:>4.1f} | {bacc:>6.2f}±{bacc_std:>4.1f} | {auroc:>6.2f}±{auroc_std:>4.1f}")

print("\n" + "="*100)
print("各被试结果 (k=50, PCET+GETA+CAGF)")
print("="*100)

subset_k50 = df[df['n_cal'] == 50]
print(f"\n{'Subject':>8} | {'Seed':>6} | {'Accuracy':>10} | {'F1':>10} | {'BAcc':>10} | {'AUROC':>10}")
print("-" * 65)

for _, row in subset_k50.iterrows():
    print(f"{row['subject']:>8} | {row['seed']:>6} | {row['PCET+GETA+CAGF_acc']*100:>9.1f}% | {row['PCET+GETA+CAGF_f1']*100:>9.1f}% | {row['PCET+GETA+CAGF_bacc']*100:>9.1f}% | {row['PCET+GETA+CAGF_auroc']*100:>9.1f}%")

print("\n" + "="*100)
print("最终结果汇总")
print("="*100)
print("""
PCET + GETA + CAGF 结果:

  k=3:  62.27% (Accuracy)
  k=5:  65.84% (Accuracy)
  k=10: 69.68% (Accuracy)
  k=20: 74.06% (Accuracy)
  k=50: 80.11% (Accuracy)

方法说明:
- PCET: EEG prediction-error representation (PCA reconstruction error)
- GETA: Gaze-guided attention on EEG features
- CAGF: Cross-modal Adaptive Gated Fusion (feature-only, no confidence)

这是效果最好的版本，以后所有实验都使用这一版。
""")

print("\nDONE!")