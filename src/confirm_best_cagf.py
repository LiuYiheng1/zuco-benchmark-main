"""
Comprehensive comparison of ALL CAGF versions to confirm the best one
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

RESULTS_DIR = "results/final"

print("="*90)
print("COMPREHENSIVE CAGF VERSION COMPARISON")
print("="*90)

k_values = [3, 5, 10, 20, 50]

files = {
    'eeg_gaze_pilot_results.csv': 'PCET+GETA+CAGF',
    'cagf_v3_cross_interaction.csv': 'CAGF_feature_only',
    'fewshot_adagtcn_proxy_comparison.csv': 'Current CAGF (wrong)',
    'cagf_comparison.csv': 'CAGF_v1/v2/v3',
}

print("\n### 1. eeg_gaze_pilot_results.csv (PCET+GETA+CAGF)")
print("-" * 70)
df1 = pd.read_csv(os.path.join(RESULTS_DIR, 'eeg_gaze_pilot_results.csv'))
if 'PCET+GETA+CAGF_acc' in df1.columns:
    for k in k_values:
        subset = df1[df1['n_cal'] == k]
        if len(subset) > 0:
            acc = subset['PCET+GETA+CAGF_acc'].mean() * 100
            print(f"  k={k}: {acc:.2f}%")

print("\n### 2. cagf_v3_cross_interaction.csv (CAGF_feature_only)")
print("-" * 70)
df2 = pd.read_csv(os.path.join(RESULTS_DIR, 'cagf_v3_cross_interaction.csv'))
if 'CAGF_feature_only_acc' in df2.columns:
    for k in k_values:
        subset = df2[df2['n_cal'] == k]
        if len(subset) > 0:
            acc = subset['CAGF_feature_only_acc'].mean() * 100
            print(f"  k={k}: {acc:.2f}%")

print("\n### 3. fewshot_adagtcn_proxy_comparison.csv (Current CAGF)")
print("-" * 70)
df3 = pd.read_csv(os.path.join(RESULTS_DIR, 'fewshot_adagtcn_proxy_comparison.csv'))
if 'PCET_GETA_CAGF_acc' in df3.columns:
    for k in k_values:
        subset = df3[df3['k'] == k]
        if len(subset) > 0:
            acc = subset['PCET_GETA_CAGF_acc'].mean() * 100
            print(f"  k={k}: {acc:.2f}%")

print("\n### 4. cagf_comparison.csv (CAGF v1/v2/v3)")
print("-" * 70)
df4 = pd.read_csv(os.path.join(RESULTS_DIR, 'cagf_comparison.csv'))
for method in ['CAGF_v1', 'CAGF_v2', 'CAGF_v3', 'CAGF_v4']:
    acc_col = f'{method}_acc'
    if acc_col in df4.columns:
        print(f"\n  {method}:")
        for k in k_values:
            subset = df4[df4['k'] == k]
            if len(subset) > 0:
                acc = subset[acc_col].mean() * 100
                print(f"    k={k}: {acc:.2f}%")

print("\n" + "="*90)
print("SIDE-BY-SIDE COMPARISON AT k=50")
print("="*90)

results_k50 = {}

if 'PCET+GETA+CAGF_acc' in df1.columns:
    results_k50['eeg_gaze_pilot (PCET+GETA+CAGF)'] = df1[df1['n_cal']==50]['PCET+GETA+CAGF_acc'].mean() * 100

if 'CAGF_feature_only_acc' in df2.columns:
    results_k50['cagf_v3 (CAGF_feature_only)'] = df2[df2['n_cal']==50]['CAGF_feature_only_acc'].mean() * 100

if 'PCET_GETA_CAGF_acc' in df3.columns:
    results_k50['fewshot (Current CAGF)'] = df3[df3['k']==50]['PCET_GETA_CAGF_acc'].mean() * 100

if 'CAGF_v3_acc' in df4.columns:
    results_k50['cagf_redesign (CAGF_v3)'] = df4[df4['k']==50]['CAGF_v3_acc'].mean() * 100

sorted_results = sorted(results_k50.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<5} | {'Method':<40} | {'k=50 Acc':>12}")
print("-" * 60)
for i, (name, acc) in enumerate(sorted_results, 1):
    print(f"{i:<5} | {name:<40} | {acc:>11.2f}%")

print("\n" + "="*90)
print("CONCLUSION")
print("="*90)
best_name, best_acc = sorted_results[0]
print(f"""
BEST VERSION: {best_name}
k=50 Accuracy: {best_acc:.2f}%

This version should be used for all future experiments.
""")

print("\n" + "="*90)
print("FULL COMPARISON ACROSS ALL SHOTS")
print("="*90)

print(f"\n{'Method':<45} | {'k=3':>8} | {'k=5':>8} | {'k=10':>8} | {'k=20':>8} | {'k=50':>8}")
print("-" * 95)

all_methods = {
    'eeg_gaze_pilot (PCET+GETA+CAGF)': (df1, 'n_cal', 'PCET+GETA+CAGF_acc'),
    'cagf_v3 (CAGF_feature_only)': (df2, 'n_cal', 'CAGF_feature_only_acc'),
    'fewshot (Current CAGF)': (df3, 'k', 'PCET_GETA_CAGF_acc'),
    'cagf_redesign (CAGF_v3)': (df4, 'k', 'CAGF_v3_acc'),
}

for name, (df, k_col, acc_col) in all_methods.items():
    vals = []
    for k in k_values:
        subset = df[df[k_col] == k]
        if len(subset) > 0 and acc_col in df.columns:
            acc = subset[acc_col].mean() * 100
            vals.append(f"{acc:>7.2f}%")
        else:
            vals.append(f"{'N/A':>8}")
    print(f"{name:<45} | {' | '.join(vals)}")

print("\nDONE!")