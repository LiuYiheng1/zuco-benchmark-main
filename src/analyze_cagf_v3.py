import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import pandas as pd

RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"

df = pd.read_csv(os.path.join(RESULTS_DIR, 'cagf_v3_cross_interaction.csv'))
shots = [3, 5, 10, 20, 50]
methods = ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only',
           'CAGF_without_confidence', 'CAGF_full_old', 'CAGF_v3_cross_interaction']

print("="*120)
print("CAGF-v3 CROSS-INTERACTION FUSION RESULTS")
print("="*120)

print(f"\n{'Method':<35}", end='')
for s in shots:
    print(f"{'S'+str(s):>14}", end='')
print()
print("-"*110)

for m in methods:
    print(f"{m:<35}", end='')
    for s in shots:
        sub = df[df['n_cal'] == s]
        if len(sub) > 0:
            v = sub[f'{m}_acc'].mean()
            sv = sub[f'{m}_acc'].std()
            print(f"{v*100:>12.2f}%±{sv*100:.1f}", end='')
        else:
            print(f"{'N/A':>14}", end='')
    print()

print("\n" + "="*120)
print("SUCCESS CRITERIA VERIFICATION")
print("="*120)

print("\n[CAGF_v3] CAGF_v3 >= CAGF_feature_only:")
pass_count = 0
for s in shots:
    sub = df[df['n_cal'] == s]
    v3 = sub['CAGF_v3_cross_interaction_acc'].mean()
    feat = sub['CAGF_feature_only_acc'].mean()
    status = "PASS" if v3 >= feat else "FAIL"
    if status == "PASS":
        pass_count += 1
    print(f"  {s}-shot: v3={v3*100:.2f}%, feature_only={feat*100:.2f}%, diff={(v3-feat)*100:.2f}% [{status}]")

print("\n[CAGF_v3] CAGF_v3 > concat:")
for s in shots:
    sub = df[df['n_cal'] == s]
    v3 = sub['CAGF_v3_cross_interaction_acc'].mean()
    concat = sub['EEG+Gaze_concat_acc'].mean()
    status = "PASS" if v3 > concat else "FAIL"
    print(f"  {s}-shot: v3={v3*100:.2f}%, concat={concat*100:.2f}%, diff={(v3-concat)*100:.2f}% [{status}]")

print("\n[CAGF_v3] CAGF_v3 > static_average:")
for s in shots:
    sub = df[df['n_cal'] == s]
    v3 = sub['CAGF_v3_cross_interaction_acc'].mean()
    static = sub['Static_average_acc'].mean()
    status = "PASS" if v3 > static else "FAIL"
    print(f"  {s}-shot: v3={v3*100:.2f}%, static={static*100:.2f}%, diff={(v3-static)*100:.2f}% [{status}]")

print(f"\nOverall: v3 >= feature_only in {pass_count}/5 shots")

report = []
report.append("# CAGF-v3 Cross-Interaction Fusion Report\n\n")
report.append("## Results\n\n")
report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
report.append("|--------|--------|--------|---------|---------|--------|\n")
for m in methods:
    row = f"| {m} |"
    for s in shots:
        sub = df[df['n_cal'] == s]
        if len(sub) > 0:
            v = sub[f'{m}_acc'].mean()
            sv = sub[f'{m}_acc'].std()
            row += f" {v*100:.1f}±{sv*100:.1f} |"
        else:
            row += " - |"
    report.append(row)

report.append("\n## Success Criteria\n\n")
report.append("| Criterion | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot | Result |\n")
report.append("|-----------|--------|--------|---------|---------|---------|--------|\n")

criteria_results = []
for crit_name, v3_col, comp_col, comparison in [
    ("v3 >= feature_only", "CAGF_v3_cross_interaction_acc", "CAGF_feature_only_acc", ">="),
    ("v3 > concat", "CAGF_v3_cross_interaction_acc", "EEG+Gaze_concat_acc", ">"),
    ("v3 > static", "CAGF_v3_cross_interaction_acc", "Static_average_acc", ">"),
]:
    row = f"| {crit_name} |"
    all_pass = True
    for s in shots:
        sub = df[df['n_cal'] == s]
        v3 = sub[v3_col].mean()
        comp = sub[comp_col].mean()
        if comparison == ">=":
            p = v3 >= comp
        else:
            p = v3 > comp
        row += f" {'PASS' if p else 'FAIL'} |"
        if not p:
            all_pass = False
    row += f" {'PASS' if all_pass else 'FAIL'} |"
    report.append(row)
    criteria_results.append(all_pass)

report.append("\n## Macro-F1 / BAcc Check\n\n")
for m in ['CAGF_v3_cross_interaction']:
    f1_pass = True
    bacc_pass = True
    for s in shots:
        sub = df[df['n_cal'] == s]
        v3_f1 = sub[f'{m}_f1'].mean()
        feat_f1 = sub['CAGF_feature_only_f1'].mean()
        v3_bacc = sub[f'{m}_bacc'].mean()
        feat_bacc = sub['CAGF_feature_only_bacc'].mean()
        if v3_f1 < feat_f1 - 0.01:
            f1_pass = False
        if v3_bacc < feat_bacc - 0.01:
            bacc_pass = False
    report.append(f"- **{m}**: F1_check=[{'PASS' if f1_pass else 'FAIL'}], BAcc_check=[{'PASS' if bacc_pass else 'FAIL'}]\n")

overall_pass = sum(criteria_results) >= 2
report.append(f"\n## Conclusion\n\n")
report.append(f"**CAGF-v3 (Cross-modal Adaptive Gated Fusion)** uses cross-modal interaction features:\n")
report.append(f"- `abs_diff = |z_eeg - z_gaze|`: disagreement magnitude between modalities\n")
report.append(f"- `hadamard = z_eeg * z_gaze`: co-activation pattern\n")
report.append(f"\n**Gate input**: `concat([z_eeg, z_gaze, abs_diff, hadamard])`\n")
report.append(f"**Alpha**: `sigmoid(MLP(gate_input))`\n")
report.append(f"**Fusion**: `z_fused = alpha * z_eeg + (1-alpha) * z_gaze`\n\n")
report.append(f"**Result**: {'SUCCESS' if overall_pass else 'NEEDS IMPROVEMENT'} - v3 {'>=' if criteria_results[0] else '<'} feature_only in {sum(criteria_results)}/3 criteria\n")

report_text = "".join(report)
with open(os.path.join(REPORTS_DIR, 'cagf_v3_report.md'), 'w') as f:
    f.write(report_text)

print(f"\nReport saved to: {REPORTS_DIR}/cagf_v3_report.md")
print("\nDone!")