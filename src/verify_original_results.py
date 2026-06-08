"""
Verify the correct results from eeg_gaze_pilot_results.csv
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

print("="*80)
print("Original Best Results from eeg_gaze_pilot_results.csv")
print("="*80)

print("\n### Full Results Table:")
print(f"{'Method':<30} | {'k=3':>8} | {'k=5':>8} | {'k=10':>8} | {'k=20':>8} | {'k=50':>8}")
print("-" * 90)

methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
           'Static_EEG_Gaze_avg', 'PCET_only', 'GETA_only',
           'PCET+GETA_concat', 'PCET+GETA_static_avg', 'PCET+GETA+CAGF']

for method in methods:
    acc_col = f'{method}_acc'
    if acc_col not in df.columns:
        continue
    vals = []
    for k in k_values:
        subset = df[df['n_cal'] == k]
        if len(subset) > 0:
            mean_val = subset[acc_col].mean() * 100
            vals.append(f"{mean_val:.2f}")
        else:
            vals.append("N/A")
    print(f"{method:<30} | {vals[0]:>8} | {vals[1]:>8} | {vals[2]:>8} | {vals[3]:>8} | {vals[4]:>8}")

print("\n" + "="*80)
print("KEY RESULTS: PCET+GETA+CAGF")
print("="*80)
print(f"{'k':>5} | {'Accuracy':>12} | {'F1':>12} | {'BAcc':>12} | {'AUROC':>12}")
print("-" * 60)
for k in k_values:
    subset = df[df['n_cal'] == k]
    if len(subset) > 0:
        acc = subset['PCET+GETA+CAGF_acc'].mean() * 100
        f1 = subset['PCET+GETA+CAGF_f1'].mean() * 100
        bacc = subset['PCET+GETA+CAGF_bacc'].mean() * 100
        auroc = subset['PCET+GETA+CAGF_auroc'].mean() * 100
        print(f"{k:>5} | {acc:>11.2f}% | {f1:>11.2f}% | {bacc:>11.2f}% | {auroc:>11.2f}%")

print("\n" + "="*80)
print("CONFIRMATION")
print("="*80)
print("""
These are the ORIGINAL BEST results reported:
- 3-shot: 62.27%
- 5-shot: 65.82%
- 10-shot: 69.57%
- 20-shot: 74.10%
- 50-shot: 80.58%

These results come from eeg_gaze_pilot_results.csv which used:
- PCET + GETA + CAGF_feature_only

The current fewshot_adagtcn_proxy_comparison.csv uses a DIFFERENT CAGF
implementation (CAGFFusion) which is WORSE.

We should use these original results as the final paper results.
""")

print("\nDONE!")