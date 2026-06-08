"""
Final comparison: CAGF_v3 (new best) vs all other methods
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

RESULTS_DIR = "results/final"

df_simple = pd.read_csv(os.path.join(RESULTS_DIR, "fewshot_adagtcn_proxy_comparison.csv"))
df_gcn = pd.read_csv(os.path.join(RESULTS_DIR, "fewshot_gcn_proxy_fixed.csv"))
df_cagf = pd.read_csv(os.path.join(RESULTS_DIR, "cagf_comparison.csv"))

k_values = [3, 5, 10, 20, 50]

print("="*80)
print("FINAL COMPARISON: New CAGF_v3 vs All Methods")
print("="*80)

methods_order = [
    ('EEG_SVM', df_simple),
    ('Gaze_SVM', df_simple),
    ('EEG_MLP', df_simple),
    ('Gaze_MLP', df_simple),
    ('Concat', df_simple),
    ('StaticAvg', df_simple),
    ('EEG_Gaze_concat', df_gcn),
    ('AdaGTCN_lite', df_gcn),
    ('PCET', df_simple),
    ('GETA', df_simple),
    ('PCET_GETA_CAGF', df_simple),
    ('CAGF_v1', df_cagf),
    ('CAGF_v2', df_cagf),
    ('CAGF_v3', df_cagf),
    ('CAGF_v4', df_cagf),
]

print("\nAccuracy (%):")
print(f"{'Method':<22} | {'k=3':>8} | {'k=5':>8} | {'k=10':>8} | {'k=20':>8} | {'k=50':>8}")
print("-" * 80)

for method, df_src in methods_order:
    acc_col = f'{method}_acc'
    if acc_col not in df_src.columns:
        continue
    vals = []
    for k in k_values:
        subset = df_src[df_src['k'] == k]
        if len(subset) > 0:
            mean_val = subset[acc_col].mean() * 100
            vals.append(f"{mean_val:.1f}")
        else:
            vals.append("N/A")
    print(f"{method:<22} | {vals[0]:>8} | {vals[1]:>8} | {vals[2]:>8} | {vals[3]:>8} | {vals[4]:>8}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

cagf_v3_k50 = df_cagf[df_cagf['k']==50]['CAGF_v3_acc'].mean() * 100
cagf_v1_k50 = df_cagf[df_cagf['k']==50]['CAGF_v1_acc'].mean() * 100
concat_k50 = df_simple[df_simple['k']==50]['Concat_acc'].mean() * 100
staticavg_k50 = df_simple[df_simple['k']==50]['StaticAvg_acc'].mean() * 100
eeg_gaze_k50 = df_gcn[df_gcn['k']==50]['EEG_Gaze_concat_acc'].mean() * 100

print(f"""
1. CAGF_v3 (Feature Concat MLP):
   - k=50: {cagf_v3_k50:.1f}%
   - 比原版 CAGF_v1 高 {cagf_v3_k50 - cagf_v1_k50:.1f}%

2. Best performers at k=50:
   - StaticAvg: {staticavg_k50:.1f}%
   - EEG_Gaze_concat: {eeg_gaze_k50:.1f}%
   - CAGF_v3: {cagf_v3_k50:.1f}%
   - Concat: {concat_k50:.1f}%

3. 分析:
   - CAGF_v3 仍然是 feature-level concat + MLP
   - 比简单的 Ridge concat (Concat) 好 1.0%
   - 但比 StaticAvg 差 1.7%

4. 结论:
   - Feature-level MLP fusion 比 output-level gating 好
   - 但 StaticAvg 这种简单方法仍然很强
   - 原因：few-shot场景下，参数少的方法更稳定
""")

print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS: CAGF_v3 vs others at k=50")
print("="*80)

print(f"""
CAGF_v3 ({cagf_v3_k50:.1f}%) vs:
  vs StaticAvg ({staticavg_k50:.1f}%): {cagf_v3_k50 - staticavg_k50:+.1f}%
  vs EEG_Gaze_concat ({eeg_gaze_k50:.1f}%): {cagf_v3_k50 - eeg_gaze_k50:+.1f}%
  vs Concat ({concat_k50:.1f}%): {cagf_v3_k50 - concat_k50:+.1f}%
  vs CAGF_v1 ({cagf_v1_k50:.1f}%): {cagf_v3_k50 - cagf_v1_k50:+.1f}%
""")

print("\nDONE!")