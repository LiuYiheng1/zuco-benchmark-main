import pandas as pd

df = pd.read_csv('results/final/eeg_gaze_pilot_results.csv')

print('=== eeg_gaze_pilot_results.csv ===')
methods = ['EEG_MLP', 'Gaze_MLP', 'PCET_only', 'GETA_only', 'EEG+Gaze_concat', 'Static_EEG_Gaze_avg', 'PCET+GETA+CAGF']

for m in methods:
    acc_col = m + '_acc'
    if acc_col in df.columns:
        vals = []
        for k in [3, 5, 10, 20, 50]:
            subset = df[df['n_cal'] == k]
            if len(subset) > 0:
                vals.append(f"{subset[acc_col].mean()*100:.1f}")
            else:
                vals.append('N/A')
        print(f'{m:20s}: {" | ".join(vals)}')