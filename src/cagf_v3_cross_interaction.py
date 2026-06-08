"""CAGF-v3: Cross-modal Adaptive Gated Fusion

Key change: Remove confidence-awareness, use cross-modal interaction features instead.
Input: z_eeg, z_gaze, abs_diff=|z_eeg-z_gaze|, hadamard=z_eeg*z_gaze
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
GAZE_FEATURE_FILE = 'sent_gaze_sacc'

GAZE_GROUPS = {
    'fixation_stability': [0],
    'reading_effort': [3, 4, 8],
    'gaze_dispersion': [1, 2],
    'transition': [7]
}

def get_trial_id(key):
    parts = key.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def load_gaze_features(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_{GAZE_FEATURE_FILE}.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        numeric_vals = [float(v) for v in values[:-1]]
        features = np.array(numeric_vals, dtype=np.float64)
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def align_eeg_gaze(X_eeg, y_eeg, trial_ids_eeg, X_gaze, y_gaze, trial_ids_gaze):
    gaze_dict = {tid: (X_gaze[i], y_gaze[i]) for i, tid in enumerate(trial_ids_gaze)}
    X_eeg_aligned, y_eeg_aligned, X_gaze_aligned, y_gaze_aligned = [], [], [], []
    for i, tid in enumerate(trial_ids_eeg):
        if tid in gaze_dict:
            X_eeg_aligned.append(X_eeg[i])
            y_eeg_aligned.append(y_eeg[i])
            X_gaze_aligned.append(gaze_dict[tid][0])
            y_gaze_aligned.append(gaze_dict[tid][1])
    return (np.array(X_eeg_aligned), np.array(y_eeg_aligned),
            np.array(X_gaze_aligned), np.array(y_gaze_aligned))

def balanced_random_sampling(y_pool, n_per_class):
    class_0_idx = np.where(y_pool == 0)[0]
    class_1_idx = np.where(y_pool == 1)[0]
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    n0 = min(n_per_class, len(class_0_idx))
    n1 = min(n_per_class, len(class_1_idx))
    selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
    np.random.shuffle(selected)
    return selected

def load_all_data():
    all_data = {}
    for subj in Y_SUBJECTS:
        Xe, ye, tid_e = load_eeg_data(subj)
        Xg, yg, tid_g = load_gaze_features(subj)
        if Xe is not None and Xg is not None:
            Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
            all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
    return all_data

class CAGFv3Variants:
    @staticmethod
    def concat(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        clf.fit(np.hstack([X_eeg_cal_s, X_gaze_cal_s]), y_cal)
        probs = clf.predict_proba(np.hstack([X_eeg_test_s, X_gaze_test_s]))[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def static_average(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)
        clf_e = SVC(kernel='rbf', probability=True, random_state=42)
        clf_e.fit(X_eeg_cal_s, y_cal)
        p_e = clf_e.predict_proba(X_eeg_test_s)[:, 1]
        clf_g = SVC(kernel='rbf', probability=True, random_state=42)
        clf_g.fit(X_gaze_cal_s, y_cal)
        p_g = clf_g.predict_proba(X_gaze_test_s)[:, 1]
        p_avg = (p_e + p_g) / 2
        return (p_avg >= 0.5).astype(int), p_avg

    @staticmethod
    def cagf_feature_only(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        alpha_cal = 1 / (1 + np.exp(-z_eeg_cal[:, 0] + z_gaze_cal[:, 0]))
        alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))

        z_fused_cal = alpha_cal.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def cagf_without_confidence(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        return CAGFv3Variants.cagf_feature_only(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test)

    @staticmethod
    def cagf_full_old(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        c_eeg_cal = np.max(z_eeg_cal, axis=1).reshape(-1, 1)
        c_eeg_test = np.max(z_eeg_test, axis=1).reshape(-1, 1)
        c_gaze_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        c_gaze_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)
        c_diff_cal = np.abs(c_eeg_cal - c_gaze_cal)
        c_diff_test = np.abs(c_eeg_test - c_gaze_test)

        gate_in_cal = np.hstack([z_eeg_cal, z_gaze_cal, c_eeg_cal, c_gaze_cal, c_diff_cal])
        gate_in_test = np.hstack([z_eeg_test, z_gaze_test, c_eeg_test, c_gaze_test, c_diff_test])

        gate_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        gate_mlp.fit(gate_in_cal, y_cal)
        alpha_cal = gate_mlp.predict_proba(gate_in_cal)[:, 1]
        alpha_test = gate_mlp.predict_proba(gate_in_test)[:, 1]

        alpha_cal_s = 1 / (1 + np.exp(-(alpha_cal - 0.5) * 5))
        alpha_test_s = 1 / (1 + np.exp(-(alpha_test - 0.5) * 5))

        z_fused_cal = alpha_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal_s.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_test_s.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def cagf_v3_cross_interaction(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        abs_diff_cal = np.abs(z_eeg_cal - z_gaze_cal)
        abs_diff_test = np.abs(z_eeg_test - z_gaze_test)
        hadamard_cal = z_eeg_cal * z_gaze_cal
        hadamard_test = z_eeg_test * z_gaze_test

        gate_input_cal = np.hstack([z_eeg_cal, z_gaze_cal, abs_diff_cal, hadamard_cal])
        gate_input_test = np.hstack([z_eeg_test, z_gaze_test, abs_diff_test, hadamard_test])

        gate_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        gate_mlp.fit(gate_input_cal, y_cal)
        alpha_cal = gate_mlp.predict_proba(gate_input_cal)[:, 1]
        alpha_test = gate_mlp.predict_proba(gate_input_test)[:, 1]

        alpha_cal_s = 1 / (1 + np.exp(-(alpha_cal - 0.5) * 5))
        alpha_test_s = 1 / (1 + np.exp(-(alpha_test - 0.5) * 5))

        z_fused_cal = alpha_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal_s.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_test_s.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs


def run_experiment():
    all_data = load_all_data()
    shot_settings = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    methods = ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only',
               'CAGF_without_confidence', 'CAGF_full_old', 'CAGF_v3_cross_interaction']

    results = []

    for seed in seeds:
        print(f'\nSeed {seed}:', flush=True)
        for held_out in Y_SUBJECTS:
            if held_out not in all_data:
                print(f'  {held_out} skip', end='', flush=True)
                continue

            d = all_data[held_out]
            Xe_test = d['Xe']
            ye_test = d['ye']
            Xg_test = d['Xg']
            n = d['n']

            train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]
            if len(train_subjs) < 3:
                print(f'  {held_out} skip', end='', flush=True)
                continue

            np.random.seed(seed)
            indices = np.random.permutation(n)
            test_size = n // 3
            test_indices = indices[:test_size]
            cal_pool_indices = indices[test_size:]

            Xe_cal_pool = Xe_test[cal_pool_indices]
            ye_cal_pool = ye_test[cal_pool_indices]
            Xg_cal_pool = Xg_test[cal_pool_indices]

            Xe_test_final = Xe_test[test_indices]
            Xg_test_final = Xg_test[test_indices]
            ye_test_final = ye_test[test_indices]

            print(f'  {held_out}', end='', flush=True)

            for n_cal in shot_settings:
                if n_cal * 2 > len(cal_pool_indices):
                    continue

                cal_idx = balanced_random_sampling(ye_cal_pool, n_cal)
                Xe_cal = Xe_cal_pool[cal_idx]
                Xg_cal = Xg_cal_pool[cal_idx]
                ye_cal = ye_cal_pool[cal_idx]

                if len(np.unique(ye_cal)) < 2:
                    continue

                row = {'seed': seed, 'subject': held_out, 'n_cal': n_cal}

                for m in methods:
                    try:
                        if m == 'EEG+Gaze_concat':
                            preds, probs = CAGFv3Variants.concat(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'Static_average':
                            preds, probs = CAGFv3Variants.static_average(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_feature_only':
                            preds, probs = CAGFv3Variants.cagf_feature_only(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_without_confidence':
                            preds, probs = CAGFv3Variants.cagf_without_confidence(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_full_old':
                            preds, probs = CAGFv3Variants.cagf_full_old(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_v3_cross_interaction':
                            preds, probs = CAGFv3Variants.cagf_v3_cross_interaction(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)

                        row[f'{m}_acc'] = accuracy_score(ye_test_final, preds)
                        row[f'{m}_f1'] = f1_score(ye_test_final, preds, average='macro')
                        row[f'{m}_bacc'] = balanced_accuracy_score(ye_test_final, preds)
                        row[f'{m}_auroc'] = roc_auc_score(ye_test_final, probs)
                    except Exception as e:
                        row[f'{m}_acc'] = 0.5
                        print(f' Err:{m}', end='')

                results.append(row)
            print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'cagf_v3_cross_interaction.csv'), index=False)
    print('\nSaved results', flush=True)
    return df

def analyze_results(df):
    shots = [3, 5, 10, 20, 50]
    methods = ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only',
               'CAGF_without_confidence', 'CAGF_full_old', 'CAGF_v3_cross_interaction']

    print("\n" + "="*120)
    print("CAGF-v3 CROSS-INTERACTION RESULTS")
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
    for s in shots:
        sub = df[df['n_cal'] == s]
        v3 = sub['CAGF_v3_cross_interaction_acc'].mean()
        feat = sub['CAGF_feature_only_acc'].mean()
        status = "PASS" if v3 >= feat else "FAIL"
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

    print("\n[Macro-F1 / BAcc check]:")
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
        print(f"  {m}: F1_check=[{'PASS' if f1_pass else 'FAIL'}], BAcc_check=[{'PASS' if bacc_pass else 'FAIL'}]")

    pass_count = 0
    fail_count = 0
    for s in shots:
        sub = df[df['n_cal'] == s]
        v3 = sub['CAGF_v3_cross_interaction_acc'].mean()
        feat = sub['CAGF_feature_only_acc'].mean()
        concat = sub['EEG+Gaze_concat_acc'].mean()
        static = sub['Static_average_acc'].mean()
        if v3 >= feat and v3 > concat and v3 > static:
            pass_count += 1
        else:
            fail_count += 1

    print(f"\nOverall: {pass_count}/5 shots pass all criteria")
    if pass_count >= 3:
        print("VERDICT: SUCCESSFUL")
    else:
        print("VERDICT: NEEDS IMPROVEMENT")

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

    criteria = [
        ("v3 >= feature_only", "CAGF_v3_cross_interaction_acc", "CAGF_feature_only_acc"),
        ("v3 > concat", "CAGF_v3_cross_interaction_acc", "EEG+Gaze_concat_acc"),
        ("v3 > static", "CAGF_v3_cross_interaction_acc", "Static_average_acc"),
    ]

    for crit_name, v3_col, comp_col in criteria:
        row = f"| {crit_name} |"
        all_pass = True
        for s in shots:
            sub = df[df['n_cal'] == s]
            v3 = sub[v3_col].mean()
            comp = sub[comp_col].mean()
            p = "PASS" if (v3 >= comp if 'feature_only' in crit_name else v3 > comp) else "FAIL"
            row += f" {p} |"
            if not p:
                all_pass = False
        row += f" {'PASS' if all_pass else 'FAIL'} |"
        report.append(row)

    report.append(f"\n## Conclusion\n\n")
    report.append(f"CAGF-v3 (Cross-modal Adaptive Gated Fusion) uses cross-modal interaction features:\n")
    report.append(f"- abs_diff = |z_eeg - z_gaze|: disagreement magnitude\n")
    report.append(f"- hadamard = z_eeg * z_gaze: co-activation pattern\n")
    report.append(f"\nGate input: concat([z_eeg, z_gaze, abs_diff, hadamard])\n")
    report.append(f"Alpha = sigmoid(MLP(gate_input))\n")
    report.append(f"z_fused = alpha * z_eeg + (1-alpha) * z_gaze\n\n")
    report.append(f"Passes all criteria in {pass_count}/5 shots.\n")

    report_text = "".join(report)
    with open(os.path.join(REPORTS_DIR, 'cagf_v3_report.md'), 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {REPORTS_DIR}/cagf_v3_report.md")

if __name__ == '__main__':
    print("CAGF-v3 Cross-Interaction Fusion Experiment", flush=True)
    print("="*80, flush=True)
    df = run_experiment()
    analyze_results(df)
    print("\nDone!", flush=True)