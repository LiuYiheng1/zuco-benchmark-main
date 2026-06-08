import pandas as pd
import numpy as np

print('=== SAGE Results Analysis ===')
sage = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/sage_results.csv')
print(f'Total rows: {len(sage)}')
print(f'Methods: {sage["method"].unique()}')
print(f'n_cal values: {sorted(sage["n_cal"].unique())}')

print('\n=== Average Accuracy by Shot and Method ===')
for n_cal in sorted(sage['n_cal'].unique()):
    print(f'\n{n_cal}-shot:')
    for method in ['LR', 'SR-GC', 'SAGE_rule']:
        data = sage[(sage['n_cal'] == n_cal) & (sage['method'] == method)]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            std = data['accuracy'].std()
            print(f'  {method}: {acc:.4f} +/- {std:.4f} (n={len(data)})')

print('\n\n=== SAGE_rule improvement over SR-GC ===')
for n_cal in sorted(sage['n_cal'].unique()):
    sr_acc = sage[(sage['n_cal'] == n_cal) & (sage['method'] == 'SR-GC')]['accuracy'].mean()
    sage_acc = sage[(sage['n_cal'] == n_cal) & (sage['method'] == 'SAGE_rule')]['accuracy'].mean()
    lr_acc = sage[(sage['n_cal'] == n_cal) & (sage['method'] == 'LR')]['accuracy'].mean()
    gap_sr = sage_acc - sr_acc
    gap_lr = sage_acc - lr_acc
    print(f'{n_cal}-shot: SAGE_rule={sage_acc:.4f} vs SR-GC={sr_acc:.4f} (gap={gap_sr:+.4f}), vs LR={lr_acc:.4f} (gap={gap_lr:+.4f})')

print('\n\n=== Win/Loss analysis ===')
for n_cal in sorted(sage['n_cal'].unique()):
    data = sage[sage['n_cal'] == n_cal]
    sage_vs_sr = []
    sage_vs_lr = []
    for subj in data['subject'].unique():
        for seed in data['seed'].unique():
            sage_acc = data[(data['subject']==subj) & (data['seed']==seed) & (data['method']=='SAGE_rule')]['accuracy'].values
            sr_acc = data[(data['subject']==subj) & (data['seed']==seed) & (data['method']=='SR-GC')]['accuracy'].values
            lr_acc = data[(data['subject']==subj) & (data['seed']==seed) & (data['method']=='LR')]['accuracy'].values
            if len(sage_acc) > 0 and len(sr_acc) > 0:
                sage_vs_sr.append(1 if sage_acc[0] > sr_acc[0] else 0)
            if len(sage_acc) > 0 and len(lr_acc) > 0:
                sage_vs_lr.append(1 if sage_acc[0] > lr_acc[0] else 0)
    print(f'{n_cal}-shot: SAGE beats SR-GC {sum(sage_vs_sr)}/{len(sage_vs_sr)} times, beats LR {sum(sage_vs_lr)}/{len(sage_vs_lr)} times')