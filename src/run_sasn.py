"""SASN Quick Test"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')

from sasn import run_experiment
import pandas as pd

print('Running SASN experiment (seed 0 only)...')
df = run_experiment()
df.to_csv('results/personalized/sasn_results.csv', index=False)

print('\nResults:')
for method in df['method'].unique():
    data = df[df['method'] == method]
    acc = data['accuracy'].mean()
    print(method + ': acc=' + str(round(acc, 4)))

print('\nSaved to results/personalized/sasn_results.csv')