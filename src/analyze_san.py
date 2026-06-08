"""Analyze SAN Results and Verify Success Criteria"""
import pandas as pd
import numpy as np

df = pd.read_csv('results/personalized/san_results.csv')

print('='*70)
print('SAN Results Analysis')
print('='*70)

shot_settings = [3, 5, 10, 20, 50]

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:')
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    source_norm = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    target_norm = df[(df['method'] == 'TargetNorm') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    san_accs = df[(df['method'] == 'SAN_ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()

    print(f'  StandardScaler: {baseline:.4f}')
    print(f'  SourceNorm: {source_norm:.4f} (gap={source_norm-baseline:+.4f})')
    print(f'  TargetNorm: {target_norm:.4f} (gap={target_norm-baseline:+.4f})')
    print(f'  ACCS: {accs:.4f} (gap={accs-baseline:+.4f})')
    print(f'  SAN_ACCS: {san_accs:.4f} (gap={san_accs-accs:+.4f})')

print('\n' + '='*70)
print('Success Criteria Verification')
print('='*70)

criterion_results = {}

criterion_1 = False
criterion_1_detail = ""
for n_cal in [5, 10]:
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    source_norm = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    gain = source_norm - baseline
    if gain >= 0.02:
        criterion_1 = True
        criterion_1_detail += f"\n  [C1] SourceNorm at {n_cal}-shot: {source_norm:.4f} >= {baseline:.4f} + 0.02 = {baseline+0.02:.4f} (gain={gain:.4f}) PASS"
    else:
        criterion_1_detail += f"\n  [C1] SourceNorm at {n_cal}-shot: {source_norm:.4f} < {baseline:.4f} + 0.02 = {baseline+0.02:.4f} (gain={gain:.4f}) FAIL"

criterion_2 = False
criterion_2_detail = ""
for n_cal in shot_settings:
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    san_accs = df[(df['method'] == 'SAN_ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    gain = san_accs - accs
    if gain >= 0.01:
        criterion_2 = True
        criterion_2_detail += f"\n  [C2] SAN_ACCS at {n_cal}-shot: {san_accs:.4f} >= {accs:.4f} + 0.01 = {accs+0.01:.4f} (gain={gain:.4f}) PASS"
    else:
        criterion_2_detail += f"\n  [C2] SAN_ACCS at {n_cal}-shot: {san_accs:.4f} < {accs:.4f} + 0.01 = {accs+0.01:.4f} (gain={gain:.4f}) FAIL"

difficult_subjects = ['YLS', 'YSL', 'YHS', 'YRP']
criterion_3 = False
criterion_3_detail = ""
for n_cal in [5, 10]:
    baseline_difficult = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal) & (df['subject'].isin(difficult_subjects))]['accuracy'].mean()
    source_norm_difficult = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal) & (df['subject'].isin(difficult_subjects))]['accuracy'].mean()
    gain = source_norm_difficult - baseline_difficult
    if gain >= 0.02:
        criterion_3 = True
        criterion_3_detail += f"\n  [C3] Difficult subjects SourceNorm at {n_cal}-shot: {source_norm_difficult:.4f} >= {baseline_difficult:.4f} + 0.02 = {baseline_difficult+0.02:.4f} (gain={gain:.4f}) PASS"
    else:
        criterion_3_detail += f"\n  [C3] Difficult subjects SourceNorm at {n_cal}-shot: {source_norm_difficult:.4f} < {baseline_difficult:.4f} + 0.02 = {baseline_difficult+0.02:.4f} (gain={gain:.4f}) FAIL"

criterion_4 = False
criterion_4_detail = ""
for n_cal in shot_settings:
    source_f1 = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal)]['macro_f1'].mean()
    source_bacc = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal)]['balanced_accuracy'].mean()
    baseline_f1 = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['macro_f1'].mean()
    baseline_bacc = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['balanced_accuracy'].mean()
    f1_gain = source_f1 - baseline_f1
    bacc_gain = source_bacc - baseline_bacc
    if f1_gain >= 0.02 and bacc_gain >= 0.02:
        criterion_4 = True
        criterion_4_detail += f"\n  [C4] Macro-F1 and BAcc both +2% at {n_cal}-shot PASS"
    else:
        criterion_4_detail += f"\n  [C4] Macro-F1 (gain={f1_gain:.4f}) and BAcc (gain={bacc_gain:.4f}) at {n_cal}-shot FAIL"

print(criterion_1_detail)
print(criterion_2_detail)
print(criterion_3_detail)
print(criterion_4_detail)

print('\n' + '='*70)
print('Summary:')
print(f'  Criterion 1 (SourceNorm >= baseline + 2%): {"PASS" if criterion_1 else "FAIL"}')
print(f'  Criterion 2 (SAN_ACCS >= ACCS + 1%): {"PASS" if criterion_2 else "FAIL"}')
print(f'  Criterion 3 (Difficult subjects +2%): {"PASS" if criterion_3 else "FAIL"}')
print(f'  Criterion 4 (Macro-F1 & BAcc sync +2%): {"PASS" if criterion_4 else "FAIL"}')
print('='*70)

criterion_results['C1'] = criterion_1
criterion_results['C2'] = criterion_2
criterion_results['C3'] = criterion_3
criterion_results['C4'] = criterion_4

print('\nDetailed Subject Analysis for Difficult Subjects:')
for subj in difficult_subjects:
    print(f'\n{subj}:')
    for n_cal in [5, 10]:
        baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal) & (df['subject'] == subj)]['accuracy'].mean()
        source = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal) & (df['subject'] == subj)]['accuracy'].mean()
        if not np.isnan(source) and not np.isnan(baseline):
            print(f'  {n_cal}-shot: StandardScaler={baseline:.4f}, SourceNorm={source:.4f} (gain={source-baseline:+.4f})')

print('\nBest Performing Methods by Shot:')
for n_cal in shot_settings:
    methods = ['StandardScaler', 'SourceNorm', 'TargetNorm', 'ACCS', 'SAN_ACCS']
    best_method = None
    best_acc = 0
    for m in methods:
        acc = df[(df['method'] == m) & (df['n_cal'] == n_cal)]['accuracy'].mean()
        if acc > best_acc:
            best_acc = acc
            best_method = m
    print(f'  {n_cal}-shot: {best_method} ({best_acc:.4f})')

if any(criterion_results.values()):
    print('\n' + '='*70)
    print('CONCLUSION: SAN PASSES SUCCESS CRITERIA!')
    print('='*70)
else:
    print('\n' + '='*70)
    print('CONCLUSION: SAN DOES NOT PASS SUCCESS CRITERIA')
    print('='*70)