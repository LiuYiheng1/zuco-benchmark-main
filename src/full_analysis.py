import pandas as pd
import numpy as np
import os

results_dir = 'd:/pycharmproject/zuco-benchmark-main/src/results'

print('='*70)
print('COMPREHENSIVE MODULE ANALYSIS')
print('='*70)

# 1. ZERO-SHOT RESULTS
print('\n### 1. ZERO-SHOT CROSS-SUBJECT (LOSO) ###')
zsl_path = os.path.join(results_dir, 'final/zero_shot_loso_results.csv')
if os.path.exists(zsl_path):
    zsl = pd.read_csv(zsl_path)
    print(f'Data shape: {zsl.shape}')
    for model in zsl['model'].unique():
        data = zsl[zsl['model'] == model]
        acc = data['accuracy'].mean()
        bacc = data['balanced_accuracy'].mean()
        print(f'  {model}: Acc={acc:.4f}, BAcc={bacc:.4f} (n={len(data)})')

# 2. SIED Lambda Sensitivity
print('\n### 2. SIED LAMBDA SENSITIVITY ###')
sied_path = os.path.join(results_dir, 'final/sied_lambda_sensitivity.csv')
if os.path.exists(sied_path):
    sied = pd.read_csv(sied_path)
    print(f'Data shape: {sied.shape}')
    for model in sied['model'].unique():
        data = sied[sied['model'] == model]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            print(f'  {model}: Acc={acc:.4f} (n={len(data)})')

# 3. SR-GC Results (PERSONALIZED)
print('\n### 3. SR-GC RESULTS (PERSONALIZED) ###')
srgc_path = os.path.join(results_dir, 'final/srgc_results.csv')
if os.path.exists(srgc_path):
    srgc = pd.read_csv(srgc_path)
    print(f'Data shape: {srgc.shape}')
    for n_cal in sorted(srgc['n_cal'].unique()):
        print(f'\n  {n_cal}-shot:')
        for method in ['EEG_SVM', 'SR-GC_a0.75_b0.25', 'SR-GC_a0.75_b0.5']:
            data = srgc[(srgc['n_cal'] == n_cal) & (srgc['method'] == method)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                std = data['accuracy'].std()
                print(f'    {method}: {acc:.4f} +/- {std:.4f}')

# 4. ACCS Results
print('\n### 4. ACCS RESULTS (PERSONALIZED) ###')
accs_path = os.path.join(results_dir, 'personalized/accs_active_calibration.csv')
if os.path.exists(accs_path):
    accs = pd.read_csv(accs_path)
    print(f'Data shape: {accs.shape}')
    for n_cal in sorted(accs['n_cal'].unique())[:5]:
        print(f'\n  {n_cal}-shot:')
        for method in ['Random', 'ACCS_KMeans']:
            data = accs[(accs['n_cal'] == n_cal) & (accs['method'] == method)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                print(f'    {method}: {acc:.4f}')

# 5. Fusion Results (Personalized)
print('\n### 5. FUSION RESULTS (PERSONALIZED) ###')
fusion_files = {
    'CV-ECF': 'personalized/cv_ecf_fusion_results.csv',
    'CLF': 'personalized/clf_logit_fusion_results.csv',
    'MACS': 'personalized/macs_fusion_results.csv',
}
for name, path in fusion_files.items():
    full_path = os.path.join(results_dir, path)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        print(f'\n{name}: shape={df.shape}')
        if 'method' in df.columns:
            for method in df['method'].unique()[:5]:
                data = df[df['method'] == method]
                if len(data) > 0 and 'accuracy' in data.columns:
                    acc = data['accuracy'].mean()
                    print(f'  {method}: {acc:.4f}')
        elif 'model' in df.columns:
            for model in df['model'].unique()[:5]:
                data = df[df['model'] == model]
                if len(data) > 0 and 'accuracy' in data.columns:
                    acc = data['accuracy'].mean()
                    print(f'  {model}: {acc:.4f}')

# 6. Domain Generalization
print('\n### 6. DOMAIN GENERALIZATION ###')
dg_files = {
    'SIED+SupCon': 'domain_generalization/sied_supcon_results.csv',
    'TCD': 'domain_generalization/tcd_full_results.csv',
}
for name, path in dg_files.items():
    full_path = os.path.join(results_dir, path)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        print(f'\n{name}: shape={df.shape}')
        for model in df['model'].unique()[:5]:
            data = df[df['model'] == model]
            if len(data) > 0 and 'accuracy' in data.columns:
                acc = data['accuracy'].mean()
                print(f'  {model}: {acc:.4f}')

# 7. Few-shot Calibration Summary
print('\n### 7. FEW-SHOT CALIBRATION SUMMARY ###')
fsc_path = os.path.join(results_dir, 'personalized/few_shot_calibration_summary.csv')
if os.path.exists(fsc_path):
    fsc = pd.read_csv(fsc_path)
    print(f'Data shape: {fsc.shape}')
    print(fsc.head(20))

print('\n' + '='*70)
print('ANALYSIS COMPLETE')
print('='*70)