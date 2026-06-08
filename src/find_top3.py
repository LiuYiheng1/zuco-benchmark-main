import pandas as pd
import numpy as np
import os

results_dir = 'd:/pycharmproject/zuco-benchmark-main/src/results'

print('='*70)
print('TOP 3 MODULES ANALYSIS')
print('='*70)

# 1. ZERO-SHOT RESULTS
print('\n### 1. ZERO-SHOT CROSS-SUBJECT (LOSO) ###')
zsl = pd.read_csv(os.path.join(results_dir, 'final/zero_shot_loso_results.csv'))
for model in zsl['model'].unique():
    data = zsl[zsl['model'] == model]
    acc = data['accuracy'].mean()
    bacc = data['balanced_accuracy'].mean()
    print(f'  {model}: Acc={acc:.4f}, BAcc={bacc:.4f}')

# 2. SIED Lambda Sensitivity
print('\n### 2. SIED Lambda Sensitivity ###')
sied = pd.read_csv(os.path.join(results_dir, 'final/sied_lambda_sensitivity.csv'))
for model in sorted(sied['model'].unique(), key=lambda x: -zsl[zsl['model']=='SIED']['accuracy'].mean() if 'SIED' in x else 0):
    data = sied[sied['model'] == model]
    if len(data) > 0:
        acc = data['accuracy'].mean()
        print(f'  {model}: Acc={acc:.4f}')

# 3. SR-GC Results
print('\n### 3. SR-GC (Personalized Few-shot) ###')
srgc = pd.read_csv(os.path.join(results_dir, 'final/srgc_results.csv'))
for n_cal in sorted(srgc['n_cal'].unique()):
    print(f'\n  {n_cal}-shot:')
    for method in ['EEG_SVM', 'SR-GC_a0.75_b0.25']:
        data = srgc[(srgc['n_cal'] == n_cal) & (srgc['method'] == method)]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            print(f'    {method}: {acc:.4f}')

# 4. ACCS Results
print('\n### 4. ACCS (Active Calibration Sampling) ###')
accs = pd.read_csv(os.path.join(results_dir, 'personalized/accs_active_calibration.csv'))
print(f'Columns: {accs.columns.tolist()}')
for n_cal in [3, 5, 10, 20]:
    print(f'\n  {n_cal}-shot per class:')
    for method in ['Random', 'ACCS_KMeans']:
        data = accs[(accs['n_cal_per_class'] == n_cal) & (accs['method'] == method)]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            print(f'    {method}: {acc:.4f}')

# 5. Fusion Methods
print('\n### 5. Fusion Methods (Personalized) ###')
for name, file in [
    ('CV-ECF', 'personalized/cv_ecf_fusion_results.csv'),
    ('MACS-Fusion', 'personalized/macs_fusion_results.csv'),
    ('SS-CMC', 'personalized/ss_cmc_results.csv'),
]:
    path = os.path.join(results_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'accuracy' in df.columns:
            if 'method' in df.columns:
                for m in df['method'].unique()[:3]:
                    d = df[df['method'] == m]
                    if len(d) > 0:
                        print(f'  {name}/{m}: {d["accuracy"].mean():.4f}')
            elif 'model' in df.columns:
                for m in df['model'].unique()[:3]:
                    d = df[df['model'] == m]
                    if len(d) > 0:
                        print(f'  {name}/{m}: {d["accuracy"].mean():.4f}')

# 6. Domain Generalization
print('\n### 6. Domain Generalization ###')
for name, file in [
    ('SIED+SupCon', 'domain_generalization/sied_supcon_results.csv'),
    ('TCD', 'domain_generalization/tcd_full_results.csv'),
]:
    path = os.path.join(results_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'accuracy' in df.columns:
            for m in df['model'].unique()[:5]:
                d = df[df['model'] == m]
                if len(d) > 0:
                    print(f'  {name}/{m}: {d["accuracy"].mean():.4f}')

# 7. SAN Results
print('\n### 7. SAN (Source-Anchored Normalization) ###')
san_files = [
    'final/san_label_free_results.csv',
    'final/san_label_free_results_v2.csv',
    'personalized/san_results.csv',
]
for f in san_files:
    path = os.path.join(results_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'method' in df.columns or 'model' in df.columns:
            col = 'method' if 'method' in df.columns else 'model'
            if 'accuracy' in df.columns:
                print(f'\n  File: {f}')
                for m in df[col].unique()[:5]:
                    d = df[df[col] == m]
                    if len(d) > 0:
                        print(f'    {m}: {d["accuracy"].mean():.4f}')

# 8. Reliability Weighting
print('\n### 8. Reliability Weighting ###')
path = os.path.join(results_dir, 'personalized/reliability_weighting_results.csv')
if os.path.exists(path):
    df = pd.read_csv(path)
    if 'accuracy' in df.columns:
        for m in df['model'].unique()[:5]:
            d = df[df['model'] == m]
            if len(d) > 0:
                print(f'  {m}: {d["accuracy"].mean():.4f}')

print('\n' + '='*70)
print('SUMMARY OF BEST METHODS')
print('='*70)