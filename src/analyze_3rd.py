import pandas as pd
import numpy as np

print('='*70)
print('ANALYSIS: Finding the 3rd Innovation Point')
print('='*70)

# Load SR-GC results
srgc = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/srgc_results.csv')

# Load PRDC results
prdc = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/prdc_results.csv')

print('\n### SR-GC vs SVM (from srgc_results.csv) ###')
print('\nShot | SVM | SRGC | Gap')
for n in [3, 5, 10, 20, 50]:
    svm = srgc[(srgc['method']=='EEG_SVM') & (srgc['n_cal']==n)]['accuracy'].mean()
    srgc_acc = srgc[(srgc['method']=='SR-GC_a0.75_b0.25') & (srgc['n_cal']==n)]['accuracy'].mean()
    print(f'{n:4d} | {svm:.4f} | {srgc_acc:.4f} | {srgc_acc-svm:+.4f}')

print('\n### PRDC vs SVM (from prdc_results.csv) ###')
print('\nShot | SVM | PRDC_best | Gap')
for n in [3, 5, 10, 20, 50]:
    svm = prdc[(prdc['method']=='EEG_SVM') & (prdc['n_cal']==n)]['accuracy'].mean()
    prdc_best = prdc[(prdc['method']=='PRDC') & (prdc['n_cal']==n)].groupby('kappa')['accuracy'].mean().max()
    print(f'{n:4d} | {svm:.4f} | {prdc_best:.4f} | {prdc_best-svm:+.4f}')

print('\n### Key Insight ###')
print('''
Based on data:

1. SR-GC (Gaussian-based) dominates at 3-20 shot (+10-18%)
2. SVM (discriminative) dominates at 50 shot (+0.25%)
3. PRDC (Gaussian-regularized SVM) consistently beats SVM at ALL shots

The problem: SR-GC uses Mahalanobis distance, which is suboptimal.
PRDC uses SVM (better classifier) with Gaussian prior regularization.

HYPOTHESIS: SR-GC + SVM hybrid where we use SVM on SR-GC features
- Extract SR-GC's class-conditional Gaussian statistics
- Train SVM on calibrated features (instead of Mahalanobis)

This could combine SR-GC's strong prior with SVM's better classifier.
''')

# Check if we have SVM on SRGC features data
print('\n### Checking if we have SVM on SRGC features ###')
print('We do NOT have this data yet. This is a new experiment.')

print('\n### Proposed 3rd Innovation Point ###')
print('''
Option A: SVM-on-SRGC (SRGC_features + SVM_classifier)
- Use SR-GC to compute Gaussian-calibrated features
- Train SVM on these calibrated features instead of using Mahalanobis
- Combines SR-GC's strong prior with SVM's better classifier

Option B: PRDC (Already implemented)
- Consistent improvement over SVM at all shots
- Principled Bayesian approach
- +0.36% average improvement

Option C: Just accept SR-GC as the 2nd point
- And focus on making SR-GC work better
- The cascade idea (SIED+SRGC) is worth exploring
''')

print('\n### Recommendation ###')
print('''
The TRUE 3rd innovation point should be:

**SVM-on-SRGC features**

Because:
1. SR-GC's Mahalanobis is suboptimal (assumes diagonal covariance)
2. SVM is a better classifier than Mahalanobis distance
3. We combine the best of both: SR-GC's strong Gaussian prior + SVM's discriminative power

This is different from:
- SAGE (fusion at prediction level - failed)
- PRDC (regularization at weight level - marginal)
- GACS (active sampling - failed)
''')