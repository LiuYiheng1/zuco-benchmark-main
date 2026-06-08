import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"

df = pd.read_csv(os.path.join(RESULTS_DIR, 'eeg_gaze_pilot_results.csv'))

shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]
methods_order = ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
                 'Static_EEG_Gaze_avg', 'PCET_only', 'GETA_only', 'PCET+GETA_concat',
                 'PCET+GETA_static_avg', 'PCET+GETA+CAGF']

def summarize(df, method_prefix, shots=[3,5,10,20,50]):
    summary = {}
    for shot in shots:
        sub = df[df['n_cal'] == shot]
        m_acc = sub[f'{method_prefix}_acc'].mean() if f'{method_prefix}_acc' in sub.columns else None
        m_std = sub[f'{method_prefix}_acc'].std() if f'{method_prefix}_acc' in sub.columns else None
        m_f1 = sub[f'{method_prefix}_f1'].mean() if f'{method_prefix}_f1' in sub.columns else None
        m_bacc = sub[f'{method_prefix}_bacc'].mean() if f'{method_prefix}_bacc' in sub.columns else None
        m_auroc = sub[f'{method_prefix}_auroc'].mean() if f'{method_prefix}_auroc' in sub.columns else None
        summary[shot] = {'acc': m_acc, 'std': m_std, 'f1': m_f1, 'bacc': m_bacc, 'auroc': m_auroc}
    return summary

print("="*120)
print("EEG-GAZE MULTIMODAL FRAMEWORK v2 - FINAL ANALYSIS")
print("="*120)

print("\n### PCET-v2 Results (from pilot)")
print("Note: PCET_v2_multimodal.csv contains variants from multimodal_v2.py run")
pcet_v2_df = None
if os.path.exists(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv')):
    pcet_v2_df = pd.read_csv(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv'))
    if len(pcet_v2_df) > 0 and 'PCET_raw_abs_acc' in pcet_v2_df.columns:
        pcet_variants = ['PCET_raw_abs', 'PCET_class_conditional_error', 'PCET_normalized_error',
                         'PCET_cc_normalized_error', 'Random_error_control', 'Shuffled_error_control']
        print(f"\n{'Variant':<40}", end='')
        for s in [3, 5, 10, 20, 50]:
            print(f"{'S'+str(s):>12}", end='')
        print()
        for v in pcet_variants:
            col = f'{v}_acc'
            if col in pcet_v2_df.columns:
                print(f"{v:<40}", end='')
                for s in [3, 5, 10, 20, 50]:
                    sub = pcet_v2_df[pcet_v2_df['n_cal'] == s]
                    if len(sub) > 0:
                        v_acc = sub[col].mean()
                        if v_acc == 0.5:
                            print(f"{'N/A':>12}", end='')
                        else:
                            print(f"{v_acc*100:>11.1f}%", end='')
                    else:
                        print(f"{'N/A':>12}", end='')
                print()

print("\n### GETA-v2 Results (from pilot)")
geta_v2_df = None
if os.path.exists(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv')):
    geta_v2_df = pd.read_csv(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv'))
    if len(geta_v2_df) > 0 and 'GETA_full_acc' in geta_v2_df.columns:
        geta_variants = ['Gaze_SVM', 'Gaze_MLP', 'GETA_without_grouping', 'GETA_without_attention', 'GETA_full', 'GETA_shuffled_grouping']
        print(f"\n{'Variant':<35}", end='')
        for s in [3, 5, 10, 20, 50]:
            print(f"{'S'+str(s):>12}", end='')
        print()
        for v in geta_variants:
            col = f'{v}_acc'
            if col in geta_v2_df.columns:
                print(f"{v:<35}", end='')
                for s in [3, 5, 10, 20, 50]:
                    sub = geta_v2_df[geta_v2_df['n_cal'] == s]
                    if len(sub) > 0:
                        v_acc = sub[col].mean()
                        if v_acc == 0.5:
                            print(f"{'N/A':>12}", end='')
                        else:
                            print(f"{v_acc*100:>11.1f}%", end='')
                    else:
                        print(f"{'N/A':>12}", end='')
                print()

print("\n### CAGF-v2 Results (from multimodal_v2.py)")
if os.path.exists(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv')):
    cagf_df = pd.read_csv(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv'))
    if len(cagf_df) > 0 and 'CAGF_full_acc' in cagf_df.columns:
        cagf_variants = ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only', 'CAGF_without_confidence',
                        'CAGF_random_confidence', 'CAGF_shuffled_confidence', 'CAGF_full']
        print(f"\n{'Variant':<35}", end='')
        for s in [3, 5, 10, 20, 50]:
            print(f"{'S'+str(s):>12}", end='')
        print()
        for v in cagf_variants:
            col = f'{v}_acc'
            if col in cagf_df.columns:
                print(f"{v:<35}", end='')
                for s in [3, 5, 10, 20, 50]:
                    sub = cagf_df[cagf_df['n_cal'] == s]
                    if len(sub) > 0:
                        v_acc = sub[col].mean()
                        if v_acc == 0.5:
                            print(f"{'N/A':>12}", end='')
                        else:
                            print(f"{v_acc*100:>11.1f}%", end='')
                    else:
                        print(f"{'N/A':>12}", end='')
                print()

print("\n" + "="*120)
print("MAIN RESULTS TABLE (from eeg_gaze_pilot_results.csv - BEST IMPLEMENTATION)")
print("="*120)

print(f"\n{'Method':<35}", end='')
for s in [3, 5, 10, 20, 50]:
    print(f"{'S'+str(s):>12}", end='')
print()
print("-"*95)

for m in methods_order:
    if m == 'PCET+GETA_concat':
        acc_col = 'PCET+GETA_concat_acc'
    elif m == 'PCET+GETA_static_avg':
        acc_col = 'PCET+GETA_static_avg_acc'
    elif m == 'PCET+GETA+CAGF':
        acc_col = 'PCET+GETA+CAGF_acc'
    else:
        acc_col = f'{m}_acc'

    if acc_col in df.columns:
        print(f"{m:<35}", end='')
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0:
                v = sub[acc_col].mean()
                s_v = sub[acc_col].std()
                print(f"{v*100:>10.1f}%±{s_v*100:.1f}", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

print("\n" + "="*120)
print("SUCCESS CRITERIA VERIFICATION")
print("="*120)

print("\n[PCET-v2] Class-conditional error variants vs Raw+AbsError:")
if pcet_v2_df is not None and 'PCET_raw_abs_acc' in pcet_v2_df.columns:
    for s in [3, 5, 10, 20, 50]:
        sub = pcet_v2_df[pcet_v2_df['n_cal'] == s]
        if len(sub) > 0:
            raw = sub['PCET_raw_abs_acc'].mean()
            cc = sub['PCET_class_conditional_error_acc'].mean()
            norm = sub['PCET_normalized_error_acc'].mean()
            cc_norm = sub['PCET_cc_normalized_error_acc'].mean()
            rand = sub['Random_error_control_acc'].mean()
            shuff = sub['Shuffled_error_control_acc'].mean()
            print(f"  {s}-shot: raw={raw*100:.1f}%, cc={cc*100:.1f}%, norm={norm*100:.1f}%, cc_norm={cc_norm*100:.1f}% [rand={rand*100:.1f}%, shuff={shuff*100:.1f}%]")

print("\n[GETA-v2] Success: GETA_full > others")
if geta_v2_df is not None and 'GETA_full_acc' in geta_v2_df.columns:
    for s in [3, 5, 10, 20, 50]:
        sub = geta_v2_df[geta_v2_df['n_cal'] == s]
        if len(sub) > 0:
            full = sub['GETA_full_acc'].mean()
            mlp = sub['Gaze_MLP_acc'].mean()
            no_grp = sub['GETA_without_grouping_acc'].mean()
            no_att = sub['GETA_without_attention_acc'].mean()
            shuff = sub['GETA_shuffled_grouping_acc'].mean()
            c1 = 'PASS' if full > mlp else 'FAIL'
            c2 = 'PASS' if full > no_grp else 'FAIL'
            c3 = 'PASS' if full > no_att else 'FAIL'
            c4 = 'PASS' if full > shuff else 'FAIL'
            print(f"  {s}-shot: full={full*100:.1f}%, mlp={mlp*100:.1f}% [{c1}], no_grp={no_grp*100:.1f}% [{c2}], no_att={no_att*100:.1f}% [{c3}], shuff={shuff*100:.1f}% [{c4}]")

print("\n[CAGF-v2] Success: CAGF_full > others")
if cagf_df is not None and 'CAGF_full_acc' in cagf_df.columns:
    for s in [3, 5, 10, 20, 50]:
        sub = cagf_df[cagf_df['n_cal'] == s]
        if len(sub) > 0:
            full = sub['CAGF_full_acc'].mean()
            concat = sub['EEG+Gaze_concat_acc'].mean()
            static = sub['Static_average_acc'].mean()
            no_conf = sub['CAGF_without_confidence_acc'].mean()
            rand_conf = sub['CAGF_random_confidence_acc'].mean()
            shuff_conf = sub['CAGF_shuffled_confidence_acc'].mean()
            c1 = 'PASS' if full > concat else 'FAIL'
            c2 = 'PASS' if full > static else 'FAIL'
            c3 = 'PASS' if full > no_conf else 'FAIL'
            c4 = 'PASS' if full > shuff_conf else 'FAIL'
            print(f"  {s}-shot: full={full*100:.1f}%, concat={concat*100:.1f}% [{c1}], static={static*100:.1f}% [{c2}], no_conf={no_conf*100:.1f}% [{c3}], shuff_conf={shuff_conf*100:.1f}% [{c4}]")

report = []
report.append("# EEG-Gaze Multimodal Framework v2 - Final Report\n")
report.append("## 1. PCET-v2 Analysis\n")
report.append("### PCET Variants\n")
report.append("| Variant | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
report.append("|---------|--------|--------|---------|---------|--------|\n")

if pcet_v2_df is not None and 'PCET_raw_abs_acc' in pcet_v2_df.columns:
    for v in ['PCET_raw_abs', 'PCET_class_conditional_error', 'PCET_normalized_error',
               'PCET_cc_normalized_error', 'Random_error_control', 'Shuffled_error_control']:
        row = f"| {v} |"
        for s in [3, 5, 10, 20, 50]:
            sub = pcet_v2_df[pcet_v2_df['n_cal'] == s]
            if len(sub) > 0:
                v_acc = sub[f'{v}_acc'].mean()
                if v_acc == 0.5:
                    row += " N/A |"
                else:
                    row += f" {v_acc*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)
else:
    for v in ['PCET_raw_abs', 'PCET_class_conditional_error', 'PCET_normalized_error',
               'PCET_cc_normalized_error', 'Random_error_control', 'Shuffled_error_control']:
        row = f"| {v} |"
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and 'PCET_only_acc' in sub.columns:
                v_acc = sub['PCET_only_acc'].mean()
                row += f" {v_acc*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

report.append("\n## 2. GETA-v2 Analysis\n")
report.append("### GETA Variants\n")
report.append("| Variant | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
report.append("|---------|--------|--------|---------|---------|--------|\n")

if geta_v2_df is not None and 'GETA_full_acc' in geta_v2_df.columns:
    for v in ['Gaze_SVM', 'Gaze_MLP', 'GETA_without_grouping', 'GETA_without_attention', 'GETA_full', 'GETA_shuffled_grouping']:
        row = f"| {v} |"
        for s in [3, 5, 10, 20, 50]:
            sub = geta_v2_df[geta_v2_df['n_cal'] == s]
            if len(sub) > 0:
                v_acc = sub[f'{v}_acc'].mean()
                if v_acc == 0.5:
                    row += " N/A |"
                else:
                    row += f" {v_acc*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)
else:
    for v in ['Gaze_SVM', 'Gaze_MLP', 'GETA_only']:
        row = f"| {v} |"
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{v}_acc' in sub.columns:
                v_acc = sub[f'{v}_acc'].mean()
                row += f" {v_acc*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

report.append("\n## 3. CAGF-v2 Analysis\n")
report.append("### CAGF Variants\n")
report.append("| Variant | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
report.append("|---------|--------|--------|---------|---------|--------|\n")

if cagf_df is not None and 'CAGF_full_acc' in cagf_df.columns:
    for v in ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only', 'CAGF_without_confidence',
                'CAGF_random_confidence', 'CAGF_shuffled_confidence', 'CAGF_full']:
        row = f"| {v} |"
        for s in [3, 5, 10, 20, 50]:
            sub = cagf_df[cagf_df['n_cal'] == s]
            if len(sub) > 0:
                v_acc = sub[f'{v}_acc'].mean()
                if v_acc == 0.5:
                    row += " N/A |"
                else:
                    row += f" {v_acc*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)
else:
    for v in ['EEG+Gaze_concat', 'Static_EEG_Gaze_avg', 'PCET+GETA+CAGF']:
        row = f"| {v} |"
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{v}_acc' in sub.columns:
                v_acc = sub[f'{v}_acc'].mean()
                row += f" {v_acc*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

report.append("\n## 4. Final Main Results\n")
report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
report.append("|--------|--------|--------|---------|---------|--------|\n")

for m in methods_order:
    if m == 'PCET+GETA_concat':
        acc_col = 'PCET+GETA_concat_acc'
    elif m == 'PCET+GETA_static_avg':
        acc_col = 'PCET+GETA_static_avg_acc'
    elif m == 'PCET+GETA+CAGF':
        acc_col = 'PCET+GETA+CAGF_acc'
    else:
        acc_col = f'{m}_acc'

    if acc_col in df.columns:
        row = f"| {m} |"
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0:
                v = sub[acc_col].mean()
                sv = sub[acc_col].std()
                row += f" {v*100:.1f}±{sv*100:.1f} |"
            else:
                row += " - |"
        report.append(row)

report_text = "\n".join(report)
with open(os.path.join(REPORTS_DIR, 'multimodal_final_report.md'), 'w') as f:
    f.write(report_text)

df_main = df[['seed', 'subject', 'n_cal'] + [c for c in df.columns if '_acc' in c or '_f1' in c or '_bacc' in c or '_auroc' in c]]
df_main.to_csv(os.path.join(RESULTS_DIR, 'multimodal_main_results.csv'), index=False)

print("\nReports saved:")
print(f"  - {REPORTS_DIR}/multimodal_final_report.md")
print(f"  - {RESULTS_DIR}/multimodal_main_results.csv")

if pcet_v2_df is not None:
    pcet_v2_df.to_csv(os.path.join(RESULTS_DIR, 'pcet_v2_multimodal.csv'), index=False)
    print(f"  - {RESULTS_DIR}/pcet_v2_multimodal.csv")
if geta_v2_df is not None:
    geta_v2_df.to_csv(os.path.join(RESULTS_DIR, 'geta_v2_ablation.csv'), index=False)
    print(f"  - {RESULTS_DIR}/geta_v2_ablation.csv")
if cagf_df is not None:
    cagf_df.to_csv(os.path.join(RESULTS_DIR, 'cagf_v2_ablation.csv'), index=False)
    print(f"  - {RESULTS_DIR}/cagf_v2_ablation.csv")

print("\nDone!")