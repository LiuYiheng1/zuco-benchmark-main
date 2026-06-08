import pandas as pd

df = pd.read_csv('results/domain_generalization/sied_supcon_results.csv')
print('Model averages:')
for m in df['model'].unique():
    acc = df[df['model']==m]['accuracy'].mean()
    std = df[df['model']==m]['accuracy'].std()
    print(f'  {m}: acc={acc:.4f}+-{std:.4f}')

sied = df[df['model']=='SIED']['accuracy'].mean()
print(f'\nSIED baseline: {sied:.4f}')
print(f'Target (SIED+1.5%): {sied+0.015:.4f}')