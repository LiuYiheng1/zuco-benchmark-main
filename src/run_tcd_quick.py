"""Quick TCD run"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import sys

print('Starting tcd_quick...')

from tcd_quick import run_raw_eeg, run_model
import pandas as pd

print('Running Raw_EEG...')
raw_results = run_raw_eeg(0)
print('Raw_EEG done: ' + str(len(raw_results)))

print('Running SIED...')
sied_results = run_model(0, 'SIED', use_conf_branch=False, use_adv=True, lambda_adv=0.01)
print('SIED done: ' + str(len(sied_results)))

print('Running TCD_full...')
tcd_results = run_model(0, 'TCD_full', use_conf_branch=True, use_corr=True, use_recon=True, use_adv=True,
                       lambda_adv=0.01, lambda_conf=0.05, lambda_corr=0.05, lambda_recon=0.01)
print('TCD_full done: ' + str(len(tcd_results)))

all_results = raw_results + sied_results + tcd_results
df = pd.DataFrame(all_results)
df.to_csv('results/domain_generalization/tcd_quick_results.csv', index=False)

print('\nResults:')
for model in df['model'].unique():
    data = df[df['model'] == model]
    acc = data['accuracy'].mean()
    print('  ' + model + ': acc=' + str(acc))

sied_acc = df[df['model'] == 'SIED']['accuracy'].mean()
tcd_acc = df[df['model'] == 'TCD_full']['accuracy'].mean()
gap = tcd_acc - sied_acc
print('\nSIED: ' + str(sied_acc))
print('TCD_full: ' + str(tcd_acc))
print('Gap: ' + str(gap))
print('Target (SIED+1.5%): ' + str(sied_acc + 0.015))

if gap >= 0.015:
    print('SUCCESS: TCD exceeds SIED by >= 1.5%')
elif gap >= 0.005:
    print('MARGINAL: TCD exceeds SIED by < 1.5%')
else:
    print('FAILED: TCD does NOT exceed SIED by 1.5%')

print('\nDone!')