import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import sys
print('Starting TCD test...', flush=True)
try:
    from tcd_disentanglement import run_sied, run_tcd_full
    print('Imports ok', flush=True)

    print('Running SIED...', flush=True)
    sied_results = run_sied(seed=0)
    print(f'SIED done: {len(sied_results)} results', flush=True)

    print('Running TCD_full...', flush=True)
    tcd_results = run_tcd_full(seed=0, lambda_adv=1.0, lambda_conf=0.5, lambda_corr=0.1, lambda_recon=0.1, lambda_supcon=0.0)
    print(f'TCD_full done: {len(tcd_results)} results', flush=True)

    import pandas as pd
    all_results = sied_results + tcd_results
    df = pd.DataFrame(all_results)
    df.to_csv('results/domain_generalization/tcd_results.csv', index=False)

    print('\nResults:')
    for model in df['model'].unique():
        data = df[df['model'] == model]
        acc = data['accuracy'].mean()
        std = data['accuracy'].std()
        print(f'  {model}: acc={acc:.4f}+-{std:.4f}')

except Exception as e:
    print(f'Error: {e}', flush=True)
    import traceback
    traceback.print_exc()

print('Done!', flush=True)