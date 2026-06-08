"""
Merge all EEG adaptation results and generate significance tests
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

RESULTS_DIR = "results/eeg_adaptation"
os.makedirs(RESULTS_DIR, exist_ok=True)

raw_df = pd.read_csv(os.path.join(RESULTS_DIR, "raw_eeg_5seeds.csv"))
coral_df = pd.read_csv(os.path.join(RESULTS_DIR, "coral_5seeds.csv"))
gaze_combined_df = pd.read_csv(os.path.join(RESULTS_DIR, "gaze_combined_5seeds.csv"))

gaze_df = gaze_combined_df[gaze_combined_df['model'] == 'Gaze_only']
combined_df = gaze_combined_df[gaze_combined_df['model'] == 'Combined']

try:
    adv_partial_df = pd.read_csv(os.path.join(RESULTS_DIR, "adversarial_partial.csv"))
    adv_01_df = adv_partial_df[adv_partial_df['model'] == 'EEG_Adversarial_lamb0.01']
    adv_05_df = adv_partial_df[adv_partial_df['model'] == 'EEG_Adversarial_lamb0.05']
    adv_10_df = adv_partial_df[adv_partial_df['model'] == 'EEG_Adversarial_lamb0.1']
    has_adversarial = True
except:
    has_adversarial = False

print("="*70)
print("EEG ADAPTATION FULL EXPERIMENT - 5 SEEDS SUMMARY")
print("="*70)

all_models = []

print("\n1. Raw EEG:")
raw_summary = raw_df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std'],
    'auroc': ['mean', 'std']
})
print(raw_summary)
all_models.append(('Raw_EEG', raw_df))

print("\n2. EEG CORAL:")
coral_summary = coral_df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std'],
    'auroc': ['mean', 'std']
})
print(coral_summary)
all_models.append(('EEG_CORAL', coral_df))

print("\n3. Gaze-only:")
gaze_summary = gaze_df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std'],
    'auroc': ['mean', 'std']
})
print(gaze_summary)
all_models.append(('Gaze_only', gaze_df))

print("\n4. Combined EEG+Gaze:")
combined_summary = combined_df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_accuracy': ['mean', 'std'],
    'auroc': ['mean', 'std']
})
print(combined_summary)
all_models.append(('Combined', combined_df))

if has_adversarial and len(adv_01_df) > 0:
    print("\n5. EEG Adversarial (partial):")
    for name, adv_df in [('λ=0.01', adv_01_df), ('λ=0.05', adv_05_df), ('λ=0.1', adv_10_df)]:
        if len(adv_df) > 0:
            print(f"  {name}: {adv_df['accuracy'].mean():.4f} +/- {adv_df['accuracy'].std():.4f} (n={len(adv_df)})")
            all_models.append((f'EEG_Adversarial_{name}', adv_df))

print("\n" + "="*70)
print("SIGNIFICANCE TESTS (Wilcoxon paired signed-rank test)")
print("="*70)

def wilcoxon_test(model1_df, model2_df, metric='accuracy'):
    merged = model1_df[['seed', 'held_out', metric]].merge(
        model2_df[['seed', 'held_out', metric]],
        on=['seed', 'held_out'],
        suffixes=('_1', '_2')
    )
    if len(merged) < 5:
        return None, None, None
    diff = merged[f'{metric}_1'] - merged[f'{metric}_2']
    if diff.abs().sum() == 0:
        return None, None, None
    stat, p = wilcoxon(diff)
    effect = diff.mean()
    return effect, p, len(merged)

significance_results = []

if has_adversarial and len(adv_01_df) >= 16:
    comparisons = [
        ('EEG_Adversarial λ=0.01', adv_01_df, 'Raw_EEG', raw_df),
        ('EEG_Adversarial λ=0.05', adv_05_df, 'Raw_EEG', raw_df),
        ('EEG_Adversarial λ=0.1', adv_10_df, 'Raw_EEG', raw_df),
        ('EEG_CORAL', coral_df, 'Raw_EEG', raw_df),
        ('EEG_Adversarial λ=0.01', adv_01_df, 'EEG_CORAL', coral_df),
        ('EEG_Adversarial λ=0.01', adv_01_df, 'Gaze_only', gaze_df),
    ]

    for name1, df1, name2, df2 in comparisons:
        if len(df1) >= 16 and len(df2) >= 16:
            mean_diff, p_val, n = wilcoxon_test(df1, df2)
            if mean_diff is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"{name1} vs {name2}: mean_diff={mean_diff:.4f}, p={p_val:.4f} {sig} (n={n})")
                significance_results.append({
                    'comparison': f'{name1} vs {name2}',
                    'mean_diff': mean_diff,
                    'p_value': p_val,
                    'n_subjects': n,
                    'significant': sig != ''
                })

sig_df = pd.DataFrame(significance_results)
if len(sig_df) > 0:
    sig_df.to_csv(os.path.join(RESULTS_DIR, "eeg_adaptation_significance.csv"), index=False)
    print(f"\nSaved significance results to {RESULTS_DIR}/eeg_adaptation_significance.csv")

merged_df = pd.concat([raw_df, coral_df, gaze_df, combined_df], ignore_index=True)
if has_adversarial:
    merged_df = pd.concat([merged_df, adv_partial_df], ignore_index=True)

merged_df.to_csv(os.path.join(RESULTS_DIR, "eeg_adaptation_full_5seed.csv"), index=False)
print(f"Merged results saved to {RESULTS_DIR}/eeg_adaptation_full_5seed.csv")

summary_data = []
for name, df in all_models:
    summary_data.append({
        'model': name,
        'accuracy_mean': df['accuracy'].mean(),
        'accuracy_std': df['accuracy'].std(),
        'macro_f1_mean': df['macro_f1'].mean(),
        'macro_f1_std': df['macro_f1'].std(),
        'balanced_accuracy_mean': df['balanced_accuracy'].mean(),
        'balanced_accuracy_std': df['balanced_accuracy'].std(),
        'auroc_mean': df['auroc'].mean(),
        'auroc_std': df['auroc'].std(),
        'n_results': len(df)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('accuracy_mean', ascending=False)
summary_df.to_csv(os.path.join(RESULTS_DIR, "eeg_adaptation_summary_5seed.csv"), index=False)

print("\n" + "="*70)
print("FINAL SUMMARY (sorted by accuracy)")
print("="*70)
for _, row in summary_df.iterrows():
    print(f"{row['model']:30s}: Acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | n={row['n_results']}")

print("\nDone!")