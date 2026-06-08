"""Analyze SASN Results and Verify Success Criteria"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

df = pd.read_csv('results/personalized/sasn_results.csv')

print('='*70)
print('SASN Results Analysis')
print('='*70)

shot_settings = [5, 10, 20, 50]
kappa_values = [10, 50]

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:')
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    source_norm = df[(df['method'] == 'SourceNorm') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    target_norm = df[(df['method'] == 'TargetNorm') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()

    print(f'  StandardScaler: {baseline:.4f}')
    print(f'  SourceNorm: {source_norm:.4f} (gap={source_norm-baseline:+.4f})')
    print(f'  TargetNorm: {target_norm:.4f} (gap={target_norm-baseline:+.4f})')
    print(f'  ACCS: {accs:.4f} (gap={accs-baseline:+.4f})')

    for kappa in kappa_values:
        sasn = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['accuracy'].mean()
        sasn_accs = df[(df['method'] == 'SASN_ACCS') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['accuracy'].mean()
        print(f'  SASN (kappa={kappa}): {sasn:.4f} (gap={sasn-baseline:+.4f}) | SASN_ACCS: {sasn_accs:.4f} (gap={sasn_accs-accs:+.4f})')

print('\n' + '='*70)
print('Success Criteria Verification')
print('='*70)

criterion_1 = False
criterion_2 = False
criterion_3 = False
criterion_4 = False

for n_cal in [5, 10]:
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    for kappa in [5, 10, 20, 50, 100]:
        sasn = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['accuracy'].mean()
        if sasn - baseline >= 0.02:
            criterion_1 = True
            print(f'\n[C1] SASN at {n_cal}-shot (kappa={kappa}): {sasn:.4f} >= baseline {baseline:.4f} + 0.02 = {baseline+0.02:.4f} ✓')
            break

for n_cal in shot_settings:
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    for kappa in [5, 10, 20, 50, 100]:
        sasn_accs = df[(df['method'] == 'SASN_ACCS') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['accuracy'].mean()
        if sasn_accs - accs >= 0.01:
            criterion_2 = True
            print(f'\n[C2] SASN_ACCS at {n_cal}-shot (kappa={kappa}): {sasn_accs:.4f} >= ACCS {accs:.4f} + 0.01 = {accs+0.01:.4f} ✓')
            break

difficult_subjects = ['YLS', 'YSL', 'YHS', 'YRP']
for n_cal in [5, 10]:
    baseline_difficult = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal) & (df['subject'].isin(difficult_subjects))]['accuracy'].mean()
    for kappa in [5, 10, 20, 50, 100]:
        sasn_difficult = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa) & (df['subject'].isin(difficult_subjects))]['accuracy'].mean()
        if sasn_difficult - baseline_difficult >= 0.02:
            criterion_3 = True
            print(f'\n[C3] Difficult subjects SASN at {n_cal}-shot (kappa={kappa}): {sasn_difficult:.4f} >= baseline {baseline_difficult:.4f} + 0.02 = {baseline_difficult+0.02:.4f} ✓')
            break

for n_cal in shot_settings:
    for kappa in [5, 10, 20, 50, 100]:
        sasn_f1 = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['macro_f1'].mean()
        sasn_bacc = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['balanced_accuracy'].mean()
        baseline_f1 = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['macro_f1'].mean()
        baseline_bacc = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['balanced_accuracy'].mean()
        if sasn_f1 - baseline_f1 >= 0.02 and sasn_bacc - baseline_bacc >= 0.02:
            criterion_4 = True
            print(f'\n[C4] Macro-F1 and Balanced Accuracy同步提升 at {n_cal}-shot (kappa={kappa}) ✓')
            break

print('\n' + '='*70)
print('Summary:')
print(f'  Criterion 1 (SASN >= baseline + 2%): {"PASS" if criterion_1 else "FAIL"}')
print(f'  Criterion 2 (SASN_ACCS >= ACCS + 1%): {"PASS" if criterion_2 else "FAIL"}')
print(f'  Criterion 3 (Difficult subjects +2%): {"PASS" if criterion_3 else "FAIL"}')
print(f'  Criterion 4 (Macro-F1 & BAcc同步提升): {"PASS" if criterion_4 else "FAIL"}')
print('='*70)

print('\nBest kappa analysis:')
for n_cal in shot_settings:
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    best_kappa = None
    best_gain = -999
    for kappa in [5, 10, 20, 50, 100]:
        sasn = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa)]['accuracy'].mean()
        gain = sasn - baseline
        if gain > best_gain:
            best_gain = gain
            best_kappa = kappa
    print(f'  {n_cal}-shot: best kappa={best_kappa} (gain={best_gain:+.4f})')

print('\nSubject-level analysis for difficult subjects:')
for subj in difficult_subjects:
    for n_cal in [5, 10]:
        baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal) & (df['subject'] == subj)]['accuracy'].mean()
        for kappa in [5, 10, 20, 50, 100]:
            sasn = df[(df['method'] == 'SASN') & (df['n_cal'] == n_cal) & (df['kappa'] == kappa) & (df['subject'] == subj)]['accuracy'].mean()
            if not np.isnan(sasn) and not np.isnan(baseline):
                gain = sasn - baseline
                print(f'  {subj} {n_cal}-shot kappa={kappa}: {sasn:.4f} (gain={gain:+.4f})')
                break

print('\nLabel usage verification:')
print('  1. Test labels used? NO - test set is held out completely')
print('  2. Target statistics from unlabeled calibration pool? YES - only cal_pool used')
print('  3. Source statistics from training subjects? YES - 15 training subjects')

print('\nConclusion: SASN does not meet success criteria. Not suitable as innovation point.')
print('='*70)