"""
Verify CAGF_feature_only results and compare with current results
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

RESULTS_DIR = "results/final"

df_cagf = pd.read_csv(os.path.join(RESULTS_DIR, 'cagf_v3_cross_interaction.csv'))
df_simple = pd.read_csv(os.path.join(RESULTS_DIR, 'fewshot_adagtcn_proxy_comparison.csv'))

k_values = [3, 5, 10, 20, 50]

print("="*80)
print("CAGF_feature_only vs Current Results Comparison")
print("="*80)

print("\n### CAGF_feature_only (Original Best) from cagf_v3_cross_interaction.csv:")
print(f"{'k':>5} | {'Accuracy':>12} | {'F1':>12} | {'BAcc':>12} | {'AUROC':>12}")
print("-" * 60)
for k in k_values:
    subset = df_cagf[df_cagf['n_cal'] == k]
    if len(subset) > 0:
        acc = subset['CAGF_feature_only_acc'].mean() * 100
        f1 = subset['CAGF_feature_only_f1'].mean() * 100
        bacc = subset['CAGF_feature_only_bacc'].mean() * 100
        auroc = subset['CAGF_feature_only_auroc'].mean() * 100
        print(f"{k:>5} | {acc:>11.2f}% | {f1:>11.2f}% | {bacc:>11.2f}% | {auroc:>11.2f}%")

print("\n### Current PCET_GETA_CAGF from fewshot_adagtcn_proxy_comparison.csv:")
print(f"{'k':>5} | {'Accuracy':>12} | {'F1':>12} | {'BAcc':>12} | {'AUROC':>12}")
print("-" * 60)
for k in k_values:
    subset = df_simple[df_simple['k'] == k]
    if len(subset) > 0:
        acc = subset['PCET_GETA_CAGF_acc'].mean() * 100
        f1 = subset['PCET_GETA_CAGF_f1'].mean() * 100
        bacc = subset['PCET_GETA_CAGF_bacc'].mean() * 100
        auroc = subset['PCET_GETA_CAGF_auroc'].mean() * 100
        print(f"{k:>5} | {acc:>11.2f}% | {f1:>11.2f}% | {bacc:>11.2f}% | {auroc:>11.2f}%")

print("\n### Difference (Current - Original):")
print(f"{'k':>5} | {'Accuracy':>12} | {'F1':>12}")
print("-" * 35)
for k in k_values:
    subset_c = df_cagf[df_cagf['n_cal'] == k]
    subset_s = df_simple[df_simple['k'] == k]
    if len(subset_c) > 0 and len(subset_s) > 0:
        acc_diff = (subset_s['PCET_GETA_CAGF_acc'].mean() - subset_c['CAGF_feature_only_acc'].mean()) * 100
        f1_diff = (subset_s['PCET_GETA_CAGF_f1'].mean() - subset_c['CAGF_feature_only_f1'].mean()) * 100
        print(f"{k:>5} | {acc_diff:>+11.2f}% | {f1_diff:>+11.2f}%")

print("\n" + "="*80)
print("KEY DIFFERENCES BETWEEN CAGF_feature_only vs Current CAGF")
print("="*80)
print("""
1. Classifier:
   - CAGF_feature_only: SVC(kernel='rbf', probability=True)
   - Current CAGF: RidgeClassifier

2. Final fusion:
   - CAGF_feature_only: MLP on fused z (hidden_layers=(16,))
   - Current CAGF: Direct probability threshold, no final MLP

3. Probability vector:
   - CAGF_feature_only: Uses full [p0, p1] probability vector
   - Current CAGF: Uses raw decision function, converts to prob

This explains why CAGF_feature_only is better!
""")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
We should use CAGF_feature_only as the final method, not the current CAGF.

The key improvements in CAGF_feature_only:
1. SVC with RBF kernel captures non-linear relationships better
2. Final MLP on fused z learns optimal combination
3. Full probability vector provides richer information

Current CAGF is simpler but performs worse.
""")

print("\nDONE!")