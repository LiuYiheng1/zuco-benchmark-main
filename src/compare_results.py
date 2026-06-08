import pandas as pd
import numpy as np

# Compare SR-GC performance between two result files
print('=== SR-GC Comparison: srgc_results.csv vs sage_results.csv ===')

srgc = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/srgc_results.csv')
sage = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/sage_results.csv')

print('\nFrom srgc_results.csv (SVM, 5 seeds, alpha varies):')
for n_cal in [3, 5, 10]:
    for method in ['SR-GC_a0.75_b0.25', 'SR-GC_a0.5_b0.25']:
        data = srgc[(srgc['n_cal'] == n_cal) & (srgc['method'] == method)]
        if len(data) > 0:
            print(f'  {n_cal}-shot {method}: {data["accuracy"].mean():.4f}')

print('\nFrom sage_results.csv (LR, alpha=0.25 fixed):')
for n_cal in [3, 5, 10]:
    data = sage[(sage['n_cal'] == n_cal) & (sage['method'] == 'SR-GC')]
    if len(data) > 0:
        print(f'  {n_cal}-shot SR-GC (alpha=0.25): {data["accuracy"].mean():.4f}')

print('\n\n=== Issue Identified ===')
print('The sage_results.csv uses alpha=0.25 for SR-GC, which is NOT optimal.')
print('The srgc_results.csv shows alpha=0.75 is best.')
print('Therefore the SAGE experiment was run with suboptimal SR-GC baseline.')
print('\nThis means:')
print('- SAGE_rule comparison is flawed (comparing to weak SR-GC)')
print('- We need to re-run SAGE with optimal alpha=0.75')

# What would SAGE_rule do if SR-GC was at optimal?
# Assume SR-GC_a0.75_b0.25 is the true baseline
print('\n\n=== If we assume optimal SR-GC (alpha=0.75) ===')
srgc_optimal = srgc[srgc['method'] == 'SR-GC_a0.75_b0.25']
for n_cal in [3, 5, 10, 20, 50]:
    opt = srgc_optimal[srgc_optimal['n_cal'] == n_cal]['accuracy'].mean()
    lr = sage[(sage['n_cal'] == n_cal) & (sage['method'] == 'LR')]['accuracy'].mean()
    sage_rule = sage[(sage['n_cal'] == n_cal) & (sage['method'] == 'SAGE_rule')]['accuracy'].mean()
    print(f'{n_cal}-shot: optimal SR-GC={opt:.4f}, LR={lr:.4f}, SAGE_rule={sage_rule:.4f}')
    if lr > opt:
        print(f'  -> LR beats optimal SR-GC by {lr-opt:.4f}')
    if sage_rule > opt:
        print(f'  -> SAGE_rule beats optimal SR-GC by {sage_rule-opt:.4f}')