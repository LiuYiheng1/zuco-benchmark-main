import pandas as pd
import numpy as np

accs = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/personalized/accs_active_calibration.csv')

print('='*70)
print('CRITICAL ANALYSIS: Label-free vs Balanced')
print('='*70)

# Compare label-free methods
print('\n### Label-Free Methods (NO test labels used) ###')
for n in [1, 3, 5, 10, 20, 50]:
    print(f'\n{n}-shot:')
    rand = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'Random_label_free')]['accuracy'].mean()
    km = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'KMeans_label_free')]['accuracy'].mean()
    print(f'  Random_label_free: {rand:.4f}')
    print(f'  KMeans_label_free: {km:.4f}')
    print(f'  Gap: {km-rand:+.4f}')

# Compare balanced methods (USES labels - LEAKY)
print('\n### Balanced Methods (USES labels - LEAKY for real deployment) ###')
for n in [1, 3, 5, 10, 20, 50]:
    print(f'\n{n}-shot:')
    rand = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'Random_balanced')]['accuracy'].mean()
    km = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'KMeans_balanced')]['accuracy'].mean()
    print(f'  Random_balanced: {rand:.4f}')
    print(f'  KMeans_balanced: {km:.4f}')
    print(f'  Gap: {km-rand:+.4f}')

print('\n' + '='*70)
print('CONCLUSION')
print('='*70)
print('''
Key Findings:

1. Label-Free ACCS (KMeans_label_free):
   - 3-shot: 0.5618 vs Random 0.5789 -> WORSE by -1.71%
   - 5-shot: 0.6165 vs Random 0.5976 -> BETTER by +1.89%
   - 10-shot: 0.6730 vs Random 0.6375 -> BETTER by +3.55%

2. Balanced ACCS (KMeans_balanced) - LEAKY:
   - 3-shot: 0.6329 vs Random 0.5751 -> BETTER by +5.78%
   - But uses TRUE LABELS for balanced sampling

CONCLUSION:
- KMeans_balanced is LEAKY (uses test labels) - cannot be used for real deployment
- KMeans_label_free only beats Random at 5-shot and above
- At 3-shot, KMeans_label_free is actually WORSE than Random!

Therefore, ACCS is NOT a strong innovation point because:
1. The "balanced" version has label leakage
2. The "label-free" version only works at 5-shot+
3. The improvement is modest (+2-4%) at best
''')