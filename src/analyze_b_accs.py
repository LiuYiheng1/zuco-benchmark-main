"""Analyze B-ACCS Results"""
import pandas as pd
import numpy as np

df = pd.read_csv('results/final/b_accs_results.csv')

print("B-ACCS Analysis")
print("="*60)

print("\nPseudo-label accuracy across all subjects:")
print(df.groupby('subject')['pseudo_acc'].first().describe())

print("\nB-ACCS by shot:")
shot_settings = [3, 5, 10, 20, 50]
for n_cal in shot_settings:
    print(f"\n{n_cal}-shot:")
    random = df[(df['method'] == 'Random') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    b60 = df[(df['method'] == 'B-ACCS_tau0.6') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    b70 = df[(df['method'] == 'B-ACCS_tau0.7') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    b80 = df[(df['method'] == 'B-ACCS_tau0.8') & (df['n_cal'] == n_cal)]['accuracy'].mean()

    print(f"  Random: {random:.4f}")
    print(f"  ACCS: {accs:.4f} (gap={accs-random:+.4f})")
    print(f"  B-ACCS tau=0.6: {b60:.4f} (gap vs ACCS={b60-accs:+.4f})")
    print(f"  B-ACCS tau=0.7: {b70:.4f} (gap vs ACCS={b70-accs:+.4f})")
    print(f"  B-ACCS tau=0.8: {b80:.4f} (gap vs ACCS={b80-accs:+.4f})")

print("\n" + "="*60)
print("Key Finding:")
print("B-ACCS fails because pseudo-label accuracy is only ~42.5%")
print("This is WORSE than random (50%), so high-confidence filtering")
print("removes correct samples and keeps wrong ones.")
print("="*60)

print("\n3/5/10-shot average:")
avg = df[df['n_cal'].isin([3,5,10])].groupby('method')['accuracy'].mean()
for method in ['Random', 'ACCS', 'B-ACCS_tau0.6', 'B-ACCS_tau0.7', 'B-ACCS_tau0.8']:
    if method in avg.index:
        print(f"  {method}: {avg[method]:.4f}")