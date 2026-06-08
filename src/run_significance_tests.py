"""
Significance tests for baseline comparison
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "loso")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

svm_df = pd.read_csv(os.path.join(RESULTS_DIR, "svm_all_features_loso.csv"))

def compute_significance(df, model1, model2):
    m1_data = df[df['model'] == model1].groupby('held_out')['accuracy'].mean()
    m2_data = df[df['model'] == model2].groupby('held_out')['accuracy'].mean()
    common_subjects = m1_data.index.intersection(m2_data.index)
    if len(common_subjects) < 5:
        return None
    m1_vals = m1_data.loc[common_subjects].values
    m2_vals = m2_data.loc[common_subjects].values
    try:
        stat, p = wilcoxon(m1_vals, m2_vals)
        mean_diff = np.mean(m1_vals - m2_vals)
        return {'model1': model1, 'model2': model2, 'mean_diff': mean_diff, 'p_value': p, 'n_subjects': len(common_subjects)}
    except:
        return None

comparisons = [
    ('SVM_Gaze_only', 'SVM_EEG_only'),
    ('SVM_Combined', 'SVM_Gaze_only'),
    ('SVM_Combined', 'SVM_EEG_only'),
    ('SVM_Gaze_only', 'Random'),
    ('SVM_Combined', 'Random'),
]

results = []
for m1, m2 in comparisons:
    res = compute_significance(svm_df, m1, m2)
    if res:
        results.append(res)
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{m1} vs {m2}: diff={res['mean_diff']:+.4f}, p={res['p_value']:.4f} {sig}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "significance_tests.csv"), index=False)
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'significance_tests.csv')}")