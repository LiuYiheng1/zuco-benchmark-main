import pandas as pd
df = pd.read_csv('results/domain_generalization/tcd_results.csv')
print('Model averages:')
for m in df['model'].unique():
    acc = df[df['model']==m]['accuracy'].mean()
    std = df[df['model']==m]['accuracy'].std()
    print(f'  {m}: acc={acc:.4f}+-{std:.4f}')

sied = df[df['model']=='SIED']['accuracy'].mean()
print(f'\nSIED baseline: {sied:.4f}')
print(f'Target (SIED+1.5%): {sied+0.015:.4f}')
print(f'Target (SIED+2.0%): {sied+0.02:.4f}')

print('\nCheck if targets are met:')
for m in df['model'].unique():
    if m == 'SIED':
        continue
    acc = df[df['model']==m]['accuracy'].mean()
    if acc >= sied + 0.02:
        print(f'  {m}: ACC ({acc:.4f}) >= SIED+2.0% ({sied+0.02:.4f}) - SUCCESS!')
    elif acc >= sied + 0.015:
        print(f'  {m}: ACC ({acc:.4f}) >= SIED+1.5% ({sied+0.015:.4f}) - MARGINAL')
    else:
        print(f'  {m}: ACC ({acc:.4f}) < SIED+1.5% - FAILED')