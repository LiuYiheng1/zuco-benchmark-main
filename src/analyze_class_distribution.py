import numpy as np
import pandas as pd
import os

SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS']

def analyze_class_distribution():
    results = []

    for subject in SUBJECTS:
        eeg_path = f'features/{subject}_electrode_features_all.npy'
        if not os.path.exists(eeg_path):
            continue

        eeg_feats = np.load(eeg_path, allow_pickle=True).item()

        nr_count = 0
        tsr_count = 0

        for key in eeg_feats.keys():
            if 'NR' in key:
                nr_count += 1
            elif 'TSR' in key:
                tsr_count += 1

        total = nr_count + tsr_count
        nr_pct = (nr_count / total) * 100 if total > 0 else 0
        tsr_pct = (tsr_count / total) * 100 if total > 0 else 0

        imbalance_ratio = tsr_count / nr_count if nr_count > 0 else float('inf')

        results.append({
            'subject': subject,
            'NR_count': nr_count,
            'TSR_count': tsr_count,
            'total': total,
            'NR_pct': nr_pct,
            'TSR_pct': tsr_pct,
            'imbalance_ratio': imbalance_ratio
        })

    df = pd.DataFrame(results)

    print("=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print("\nFull dataset class distribution:")
    print(df.to_string())

    print("\n\nSummary Statistics:")
    print(f"  Total samples across all subjects: {df['total'].sum()}")
    print(f"  Average NR percentage: {df['NR_pct'].mean():.2f}%")
    print(f"  Min NR percentage: {df['NR_pct'].min():.2f}% ({df.loc[df['NR_pct'].idxmin(), 'subject']})")
    print(f"  Max NR percentage: {df['NR_pct'].max():.2f}% ({df.loc[df['NR_pct'].idxmax(), 'subject']})")

    print("\n\nSubjects with severe imbalance (NR < 10%):")
    severe = df[df['NR_pct'] < 10]
    for _, row in severe.iterrows():
        print(f"  {row['subject']}: {row['NR_count']} NR vs {row['TSR_count']} TSR ({row['NR_pct']:.1f}% NR)")

    print("\n\nk-shot calibration impact analysis:")
    for k in [3, 5, 10, 20, 50]:
        print(f"\n  k={k}:")
        for _, row in df.iterrows():
            nr_avail = row['NR_count']
            tsr_avail = row['TSR_count']

            if k <= min(nr_avail, tsr_avail):
                cal_per_class = k
                feasible = "OK"
            elif k <= nr_avail:
                cal_per_class = k
                feasible = f"NR ok, but TSR has only {tsr_avail}"
            elif k <= tsr_avail:
                cal_per_class = min(k, nr_avail)
                feasible = f"NR limited to {nr_avail}, TSR has {tsr_avail}"
            else:
                cal_per_class = min(nr_avail, tsr_avail)
                feasible = f"Both limited! NR={nr_avail}, TSR={tsr_avail}"

            print(f"    {row['subject']}: {feasible}")

    return df

df = analyze_class_distribution()

print("\n\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("""
The severe class imbalance explains the Gaze_MLP performance:

1. YAK: Only 5 NR samples out of 171 (2.9% NR)
   - With k=3, calibration gets 3 NR + 3 TSR
   - Model learns almost exclusively from TSR pattern
   - Test set has 5 NR vs 166 TSR
   - Model predicts majority class -> high accuracy but wrong!

2. YAG: 16 NR vs 177 TSR (8.3% NR)
   - Similar issue

3. YFS: 14 NR vs 147 TSR (8.7% NR)
   - Similar issue

For proper few-shot learning with imbalanced data, options include:
1. Use stratified k-shot (already done)
2. Oversample minority class
3. Use class weights in loss function
4. Report balanced accuracy instead of raw accuracy
5. Use different k values for each class
""")