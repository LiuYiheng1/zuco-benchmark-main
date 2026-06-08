import pandas as pd
import numpy as np

print('='*70)
print('DETAILED ACCS VERIFICATION')
print('='*70)

# Load ACCS data
accs = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/personalized/accs_active_calibration.csv')
print(f'\nACCS data shape: {accs.shape}')
print(f'Columns: {accs.columns.tolist()}')
print(f'\nMethods: {accs["method"].unique()}')
print(f'Shots: {sorted(accs["n_cal_per_class"].unique())}')

# Compare all methods at each shot
print('\n' + '='*70)
print('ALL METHODS BY SHOT')
print('='*70)

for n in sorted(accs['n_cal_per_class'].unique()):
    print(f'\n### {n}-shot per class ###')
    data = accs[accs['n_cal_per_class'] == n]
    for method in sorted(data['method'].unique()):
        d = data[data['method'] == method]
        acc = d['accuracy'].mean()
        std = d['accuracy'].std()
        n_samples = len(d)
        print(f'  {method}: {acc:.4f} +/- {std:.4f} (n={n_samples})')

# Calculate ACCS vs Random improvement
print('\n' + '='*70)
print('ACCS vs Random improvement (balanced)')
print('='*70)

for n in sorted(accs['n_cal_per_class'].unique()):
    data = accs[accs['n_cal_per_class'] == n]
    rand = data[data['method'] == 'Random_balanced']['accuracy'].mean()
    accs_k = data[data['method'] == 'KMeans_balanced']['accuracy'].mean()
    gap = accs_k - rand
    print(f'{n}-shot: ACCS={accs_k:.4f}, Random={rand:.4f}, Gap={gap:+.4f}')

# Check if there's a Protocol A vs Protocol B difference
print('\n' + '='*70)
print('Protocol Analysis')
print('='*70)

if 'protocol' in accs.columns:
    print(f'Protocols: {accs["protocol"].unique()}')
    for proto in accs['protocol'].unique():
        print(f'\n{proto}:')
        data = accs[accs['protocol'] == proto]
        for n in [3, 5, 10]:
            d = data[data['n_cal_per_class'] == n]
            for method in ['Random_balanced', 'KMeans_balanced']:
                m = d[d['method'] == method]
                if len(m) > 0:
                    print(f'  {n}-shot {method}: {m["accuracy"].mean():.4f}')