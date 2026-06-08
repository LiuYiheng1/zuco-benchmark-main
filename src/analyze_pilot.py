import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv('results/final/eeg_gaze_pilot_results.csv')

acc_cols = [c for c in df.columns if c.endswith('_acc')]
summary = {}
for col in acc_cols:
    method = col.replace('_acc', '')
    summary[method] = {}
    for shot in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == shot][col]
        if len(sub) > 0:
            summary[method][shot] = (sub.mean(), sub.std())

print("=" * 140)
print("EEG-GAZE MULTIMODAL FRAMEWORK PILOT - MAIN RESULTS")
print("=" * 140)
header = f"{'Method':<30}" + "".join([f"{'SHOT-'+str(s):>22}" for s in [3, 5, 10, 20, 50]])
print(header)
print("-" * 140)

for method in ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
               'Static_EEG_Gaze_avg', 'PCET_only', 'GETA_only', 'PCET+GETA_concat',
               'PCET+GETA_static_avg', 'PCET+GETA+CAGF']:
    if method in summary:
        row_str = f"{method:<30}"
        for shot in [3, 5, 10, 20, 50]:
            if shot in summary[method]:
                m, s = summary[method][shot]
                row_str += f"{m*100:>19.2f}%±{s*100:.1f}%"
            else:
                row_str += f"{'N/A':>22}"
        print(row_str)

print("\n" + "=" * 140)
print("SUCCESS CRITERIA VERIFICATION")
print("=" * 140)

cagf_col = 'PCET+GETA+CAGF_acc'
concat_col = 'PCET+GETA_concat_acc'
static_col = 'PCET+GETA_static_avg_acc'
pcet_col = 'PCET_only_acc'
geta_col = 'GETA_only_acc'
gaze_mlp_col = 'Gaze_MLP_acc'
eeg_gaze_concat_col = 'EEG+Gaze_concat_acc'

print("\n[GETA] GETA > Gaze_MLP (by avg across shots):")
geta_overall = np.mean([summary['GETA_only'][s][0] for s in [3, 5, 10, 20, 50] if s in summary['GETA_only']])
gaze_mlp_overall = np.mean([summary['Gaze_MLP'][s][0] for s in [3, 5, 10, 20, 50] if s in summary['Gaze_MLP']])
print(f"  GETA avg: {geta_overall*100:.2f}%, Gaze_MLP avg: {gaze_mlp_overall*100:.2f}%, diff: {(geta_overall-gaze_mlp_overall)*100:.2f}%")
print(f"  Per-shot:")
for shot in [3, 5, 10, 20, 50]:
    sub = df[df['n_cal'] == shot]
    if len(sub) > 0:
        g = sub[geta_col].mean()
        gm = sub[gaze_mlp_col].mean()
        status = "PASS" if g > gm + 0.01 else "FAIL"
        print(f"    {shot}-shot: GETA={g*100:.2f}%, Gaze_MLP={gm*100:.2f}%, diff={(g-gm)*100:.2f}% [{status}]")

print("\n[CAGF] CAGF > concat AND CAGF > static_avg:")
cagf_overall = np.mean([summary['PCET+GETA+CAGF'][s][0] for s in [3, 5, 10, 20, 50] if s in summary['PCET+GETA+CAGF']])
concat_overall = np.mean([summary['PCET+GETA_concat'][s][0] for s in [3, 5, 10, 20, 50] if s in summary['PCET+GETA_concat']])
static_overall = np.mean([summary['PCET+GETA_static_avg'][s][0] for s in [3, 5, 10, 20, 50] if s in summary['PCET+GETA_static_avg']])
print(f"  CAGF avg: {cagf_overall*100:.2f}%, concat avg: {concat_overall*100:.2f}%, static avg: {static_overall*100:.2f}%")
for shot in [3, 5, 10, 20, 50]:
    sub = df[df['n_cal'] == shot]
    if len(sub) > 0:
        c = sub[cagf_col].mean()
        co = sub[concat_col].mean()
        st = sub[static_col].mean()
        c1 = "PASS" if c > co else "FAIL"
        c2 = "PASS" if c > st else "FAIL"
        print(f"  {shot}-shot: CAGF={c*100:.2f}%, concat={co*100:.2f}%, static={st*100:.2f}% [{c1},{c2}]")

print("\n[Full] CAGF > PCET_only AND CAGF > GETA_only AND CAGF > EEG+Gaze_concat:")
for shot in [3, 5, 10, 20, 50]:
    sub = df[df['n_cal'] == shot]
    if len(sub) > 0:
        c = sub[cagf_col].mean()
        p = sub[pcet_col].mean()
        g = sub[geta_col].mean()
        eg = sub[eeg_gaze_concat_col].mean()
        c1 = "PASS" if c > p else "FAIL"
        c2 = "PASS" if c > g else "FAIL"
        c3 = "PASS" if c > eg else "FAIL"
        print(f"  {shot}-shot: Full={c*100:.2f}%, PCET={p*100:.2f}%, GETA={g*100:.2f}%, concat={eg*100:.2f}% [{c1},{c2},{c3}]")

print("\n" + "=" * 140)
print("PILOT VERDICT")
print("=" * 140)
pass_count = 0
fail_count = 0

for shot in [3, 5, 10, 20, 50]:
    sub = df[df['n_cal'] == shot]
    if len(sub) > 0:
        c = sub[cagf_col].mean()
        p = sub[pcet_col].mean()
        g = sub[geta_col].mean()
        eg = sub[eeg_gaze_concat_col].mean()
        co = sub[concat_col].mean()
        st = sub[static_col].mean()
        gm = sub[gaze_mlp_col].mean()
        ga = sub[geta_col].mean()

        if c > p and c > g and c > eg and c > co and c > st and ga > gm:
            pass_count += 1
        else:
            fail_count += 1

print(f"Shots passing ALL criteria: {pass_count}/5")
if pass_count >= 3:
    print("RESULT: PILOT SUCCESSFUL -路线可行")
else:
    print("RESULT: 需要改进")