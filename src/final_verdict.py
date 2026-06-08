import pandas as pd
import numpy as np

print('='*70)
print('FINAL VERDICT: What Actually Works')
print('='*70)

# Load all relevant data
accs = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/personalized/accs_active_calibration.csv')
srgc = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/srgc_results.csv')
zsl = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/zero_shot_loso_results.csv')

print('\n### 1. ZERO-SHOT (LOSO) ###')
for model in ['Raw_EEG', 'SIED']:
    d = zsl[zsl['model'] == model]
    print(f'{model}: {d["accuracy"].mean():.4f}')

print('\n### 2. PERSONALIZED: SR-GC vs EEG_SVM ###')
for n in [3, 5, 10, 20, 50]:
    svm = srgc[(srgc['n_cal'] == n) & (srgc['method'] == 'EEG_SVM')]['accuracy'].mean()
    srgc_best = srgc[(srgc['n_cal'] == n) & (srgc['method'] == 'SR-GC_a0.75_b0.25')]['accuracy'].mean()
    gap = srgc_best - svm
    print(f'{n}-shot: SVM={svm:.4f}, SR-GC={srgc_best:.4f} (gap={gap:+.4f})')

print('\n### 3. PERSONALIZED: ACCS vs Random (Label-Free) ###')
for n in [1, 3, 5, 10, 20, 50]:
    rand = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'Random_label_free')]['accuracy'].mean()
    km = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'KMeans_label_free')]['accuracy'].mean()
    gap = km - rand
    print(f'{n}-shot: Random={rand:.4f}, ACCS={km:.4f} (gap={gap:+.4f})')

print('\n' + '='*70)
print('FINAL VERDICT')
print('='*70)

print('''
### What Actually Works ###

1. SIED (Zero-shot):
   - 52.86% vs Raw_EEG 50.82%
   - +2.04% improvement
   - WORKS: Can be innovation point #1

2. SR-GC (Personalized Low-shot):
   - 3-shot: 59.25% vs SVM 43.59% (+15.66%)
   - 5-shot: 59.88% vs SVM 41.56% (+18.32%)
   - WORKS: Can be innovation point #2

3. ACCS (Personalized):
   - 5-shot: 61.65% vs Random 59.76% (+1.89%)
   - 10-shot: 67.30% vs Random 63.75% (+3.55%)
   - MARGINAL: Only beats Random at 5-shot+
   - DOES NOT WORK at 1-3 shot (worse than Random!)

### Conclusion ###

TOP 2 INNOVATION POINTS:
1. SIED (Zero-shot): +2.04% over Raw_EEG
2. SR-GC (Low-shot): +15-18% over SVM at 3-5 shot

ACCS IS NOT A STRONG INNOVATION POINT because:
- At 1-3 shot, ACCS is WORSE than Random!
- At 5-shot+, improvement is marginal (+2-4%)
- SR-GC already beats SVM significantly at these settings
''')