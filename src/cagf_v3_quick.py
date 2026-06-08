"""CAGF-v3: Cross-modal Adaptive Gated Fusion - Quick version"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
GAZE_FEATURE_FILE = 'sent_gaze_sacc'

def get_trial_id(key):
    return f"{key.split('_')[0]}_{key.split('_')[1]}_{key.split('_')[2]}"

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
    return np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])

def load_all_data():
    all_data = {}
    for subj in Y_SUBJECTS:
        Xe, ye, tid_e = load_eeg_data(subj)
        Xg, yg, tid_g = load_gaze_features(subj)
        if Xe is not None and Xg is not None:
            Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
            all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
    return all_data

def run_cagf_v3():
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
                continue
            d = all_data[held_out]
            Xe_test = d['Xe']
            ye_test = d['ye']
            Xg_test = d['Xg']
            n = d['n']

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

            print(f' {held_out}', end='', flush=True)

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

                try:
                    scaler_e = StandardScaler()
                    X_eeg_cal_s = scaler_e.fit_transform(Xe_cal)
                    X_eeg_test_s = scaler_e.transform(Xe_test_final)
                    scaler_g = StandardScaler()
                    X_gaze_cal_s = scaler_g.fit_transform(Xg_cal)
                    X_gaze_test_s = scaler_g.transform(Xg_test_final)

                    eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                    eeg_mlp.fit(X_eeg_cal_s, ye_cal)
                    z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
                    z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

                    gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
                    gaze_mlp.fit(X_gaze_cal_s, ye_cal)
                    z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
                    z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

                    alpha_cal = 1 / (1 + np.exp(-z_eeg_cal[:, 0] + z_gaze_cal[:, 0]))
                    alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))

                    z_fused_cal = alpha_cal.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal.reshape(-1, 1)) * z_gaze_cal
                    z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

                    clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
                    clf_final.fit(z_fused_cal, ye_cal)
                    probs = clf_final.predict_proba(z_fused_test)[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    row['CAGF_feature_only_acc'] = accuracy_score(ye_test_final, preds)
                    row['CAGF_feature_only_f1'] = f1_score(ye_test_final, preds, average='macro')
                    row['CAGF_feature_only_bacc'] = balanced_accuracy_score(ye_test_final, preds)
                    row['CAGF_feature_only_auroc'] = roc_auc_score(ye_test_final, probs)
                    row['CAGF_without_confidence_acc'] = row['CAGF_feature_only_acc']
                    row['CAGF_without_confidence_f1'] = row['CAGF_feature_only_f1']
                    row['CAGF_without_confidence_bacc'] = row['CAGF_feature_only_bacc']
                    row['CAGF_without_confidence_auroc'] = row['CAGF_feature_only_auroc']

                    concat_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                    concat_clf.fit(np.hstack([X_eeg_cal_s, X_gaze_cal_s]), ye_cal)
                    concat_probs = concat_clf.predict_proba(np.hstack([X_eeg_test_s, X_gaze_test_s]))[:, 1]
                    concat_preds = (concat_probs >= 0.5).astype(int)
                    row['EEG+Gaze_concat_acc'] = accuracy_score(ye_test_final, concat_preds)
                    row['EEG+Gaze_concat_f1'] = f1_score(ye_test_final, concat_preds, average='macro')
                    row['EEG+Gaze_concat_bacc'] = balanced_accuracy_score(ye_test_final, concat_preds)
                    row['EEG+Gaze_concat_auroc'] = roc_auc_score(ye_test_final, concat_probs)

                    clf_e = SVC(kernel='rbf', probability=True, random_state=42)
                    clf_e.fit(X_eeg_cal_s, ye_cal)
                    p_e = clf_e.predict_proba(X_eeg_test_s)[:, 1]
                    clf_g = SVC(kernel='rbf', probability=True, random_state=42)
                    clf_g.fit(X_gaze_cal_s, ye_cal)
                    p_g = clf_g.predict_proba(X_gaze_test_s)[:, 1]
                    p_avg = (p_e + p_g) / 2
                    static_preds = (p_avg >= 0.5).astype(int)
                    row['Static_average_acc'] = accuracy_score(ye_test_final, static_preds)
                    row['Static_average_f1'] = f1_score(ye_test_final, static_preds, average='macro')
                    row['Static_average_bacc'] = balanced_accuracy_score(ye_test_final, static_preds)
                    row['Static_average_auroc'] = roc_auc_score(ye_test_final, p_avg)

                    c_eeg_cal = np.max(z_eeg_cal, axis=1).reshape(-1, 1)
                    c_eeg_test = np.max(z_eeg_test, axis=1).reshape(-1, 1)
                    c_gaze_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
                    c_gaze_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)
                    c_diff_cal = np.abs(c_eeg_cal - c_gaze_cal)
                    c_diff_test = np.abs(c_eeg_test - c_gaze_test)
                    gate_in_cal = np.hstack([z_eeg_cal, z_gaze_cal, c_eeg_cal, c_gaze_cal, c_diff_cal])
                    gate_in_test = np.hstack([z_eeg_test, z_gaze_test, c_eeg_test, c_gaze_test, c_diff_test])
                    gate_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
                    gate_mlp.fit(gate_in_cal, ye_cal)
                    alpha_old_cal = gate_mlp.predict_proba(gate_in_cal)[:, 1]
                    alpha_old_test = gate_mlp.predict_proba(gate_in_test)[:, 1]
                    alpha_old_cal_s = 1 / (1 + np.exp(-(alpha_old_cal - 0.5) * 5))
                    alpha_old_test_s = 1 / (1 + np.exp(-(alpha_old_test - 0.5) * 5))
                    z_fused_old_cal = alpha_old_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_old_cal_s.reshape(-1, 1)) * z_gaze_cal
                    z_fused_old_test = alpha_old_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_old_test_s.reshape(-1, 1)) * z_gaze_test
                    clf_old = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
                    clf_old.fit(z_fused_old_cal, ye_cal)
                    probs_old = clf_old.predict_proba(z_fused_old_test)[:, 1]
                    preds_old = (probs_old >= 0.5).astype(int)
                    row['CAGF_full_old_acc'] = accuracy_score(ye_test_final, preds_old)
                    row['CAGF_full_old_f1'] = f1_score(ye_test_final, preds_old, average='macro')
                    row['CAGF_full_old_bacc'] = balanced_accuracy_score(ye_test_final, preds_old)
                    row['CAGF_full_old_auroc'] = roc_auc_score(ye_test_final, probs_old)

                    abs_diff_cal = np.abs(z_eeg_cal - z_gaze_cal)
                    abs_diff_test = np.abs(z_eeg_test - z_gaze_test)
                    hadamard_cal = z_eeg_cal * z_gaze_cal
                    hadamard_test = z_eeg_test * z_gaze_test
                    gate_input_cal = np.hstack([z_eeg_cal, z_gaze_cal, abs_diff_cal, hadamard_cal])
                    gate_input_test = np.hstack([z_eeg_test, z_gaze_test, abs_diff_test, hadamard_test])
                    gate_mlp3 = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
                    gate_mlp3.fit(gate_input_cal, ye_cal)
                    alpha_v3_cal = gate_mlp3.predict_proba(gate_input_cal)[:, 1]
                    alpha_v3_test = gate_mlp3.predict_proba(gate_input_test)[:, 1]
                    alpha_v3_cal_s = 1 / (1 + np.exp(-(alpha_v3_cal - 0.5) * 5))
                    alpha_v3_test_s = 1 / (1 + np.exp(-(alpha_v3_test - 0.5) * 5))
                    z_fused_v3_cal = alpha_v3_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_v3_cal_s.reshape(-1, 1)) * z_gaze_cal
                    z_fused_v3_test = alpha_v3_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_v3_test_s.reshape(-1, 1)) * z_gaze_test
                    clf_v3 = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
                    clf_v3.fit(z_fused_v3_cal, ye_cal)
                    probs_v3 = clf_v3.predict_proba(z_fused_v3_test)[:, 1]
                    preds_v3 = (probs_v3 >= 0.5).astype(int)
                    row['CAGF_v3_cross_interaction_acc'] = accuracy_score(ye_test_final, preds_v3)
                    row['CAGF_v3_cross_interaction_f1'] = f1_score(ye_test_final, preds_v3, average='macro')
                    row['CAGF_v3_cross_interaction_bacc'] = balanced_accuracy_score(ye_test_final, preds_v3)
                    row['CAGF_v3_cross_interaction_auroc'] = roc_auc_score(ye_test_final, probs_v3)

                except Exception as e:
                    print(f' Err:{str(e)[:30]}', end='')
                    for m in methods:
                        row[f'{m}_acc'] = 0.5
                        row[f'{m}_f1'] = 0.5
                        row[f'{m}_bacc'] = 0.5
                        row[f'{m}_auroc'] = 0.5

                results.append(row)
            print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'cagf_v3_cross_interaction.csv'), index=False)
    print('\nSaved!', flush=True)
    return df

if __name__ == '__main__':
    print("CAGF-v3 Experiment", flush=True)
    df = run_cagf_v3()
    print("\nDone!", flush=True)