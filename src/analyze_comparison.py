"""
PCET+GETA+CAGF vs Baselines 对比分析
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

k_values = [3, 5, 10, 20, 50]

print("="*80)
print("PCET+GETA+CAGF vs Baselines 对比分析")
print("="*80)

def compare_at_k(df, method1, method2, k, metric='acc'):
    col1 = f'{method1}_{metric}'
    col2 = f'{method2}_{metric}'
    subset = df[df['k'] == k]
    if len(subset) == 0:
        return None, None
    vals1 = subset[col1].values
    vals2 = subset[col2].values
    return vals1, vals2

print("\n### PCET+GETA+CAGF vs Concat (Ridge concat)")
print(f"{'k':>5} | {'PCET_GETA_CAGF':>15} | {'Concat':>15} | {'Diff':>10} | {'Better':>10}")
print("-" * 70)

diffs_concat = []
for k in k_values:
    vals1, vals2 = compare_at_k(df_simple, 'PCET_GETA_CAGF', 'Concat', k, 'acc')
    if vals1 is not None:
        mean1 = np.mean(vals1) * 100
        mean2 = np.mean(vals2) * 100
        diff = mean1 - mean2
        diffs_concat.append(diff)
        better = "PCET" if diff > 0 else "Concat"
        print(f"{k:>5} | {mean1:>14.1f}% | {mean2:>14.1f}% | {diff:>+9.1f}% | {better:>10}")

print("\n### PCET+GETA+CAGF vs StaticAvg (Simple Average)")
print(f"{'k':>5} | {'PCET_GETA_CAGF':>15} | {'StaticAvg':>15} | {'Diff':>10} | {'Better':>10}")
print("-" * 70)

diffs_static = []
for k in k_values:
    vals1, vals2 = compare_at_k(df_simple, 'PCET_GETA_CAGF', 'StaticAvg', k, 'acc')
    if vals1 is not None:
        mean1 = np.mean(vals1) * 100
        mean2 = np.mean(vals2) * 100
        diff = mean1 - mean2
        diffs_static.append(diff)
        better = "PCET" if diff > 0 else "StaticAvg"
        print(f"{k:>5} | {mean1:>14.1f}% | {mean2:>14.1f}% | {diff:>+9.1f}% | {better:>10}")

print("\n### PCET vs EEG_SVM (看看PCET单独效果)")
print(f"{'k':>5} | {'PCET':>15} | {'EEG_SVM':>15} | {'Diff':>10} | {'Better':>10}")
print("-" * 70)

for k in k_values:
    vals1, vals2 = compare_at_k(df_simple, 'PCET', 'EEG_SVM', k, 'acc')
    if vals1 is not None:
        mean1 = np.mean(vals1) * 100
        mean2 = np.mean(vals2) * 100
        diff = mean1 - mean2
        better = "PCET" if diff > 0 else "EEG_SVM"
        print(f"{k:>5} | {mean1:>14.1f}% | {mean2:>14.1f}% | {diff:>+9.1f}% | {better:>10}")

print("\n### GETA vs EEG_SVM (看看GETA单独效果)")
print(f"{'k':>5} | {'GETA':>15} | {'EEG_SVM':>15} | {'Diff':>10} | {'Better':>10}")
print("-" * 70)

for k in k_values:
    vals1, vals2 = compare_at_k(df_simple, 'GETA', 'EEG_SVM', k, 'acc')
    if vals1 is not None:
        mean1 = np.mean(vals1) * 100
        mean2 = np.mean(vals2) * 100
        diff = mean1 - mean2
        better = "GETA" if diff > 0 else "EEG_SVM"
        print(f"{k:>5} | {mean1:>14.1f}% | {mean2:>14.1f}% | {diff:>+9.1f}% | {better:>10}")

print("\n" + "="*80)
print("关键发现")
print("="*80)
print(f"""
1. PCET+GETA+CAGF 在所有k值下都**不如** StaticAvg 和 Concat
   - 平均比 Concat 差: {np.mean(diffs_concat):.1f}%
   - 平均比 StaticAvg 差: {np.mean(diffs_static):.1f}%

2. PCET 本身和 EEG_SVM 效果几乎一样
   - 说明 PCET 的 prediction-error 特征并没有带来额外信息

3. GETA 本身和 EEG_SVM 效果也几乎一样
   - 说明 GETA 的 gaze-guided attention 也没有带来额外提升

4. 问题根源：
   - CAGF 的门控机制过于简单，当两个模型预测相似时等价于平均
   - PCET 的 PCA 重建误差在 few-shot 场景下区分度不够
   - GETA 的 attention 机制可能没有真正学到有意义的权重

5. 建议：
   - 需要重新设计 CAGF 机制，使其在 few-shot 下更鲁棒
   - 或者重新思考创新点：简单 concat 已经足够好
   - 可能需要更大的数据集来验证创新点的价值
""")