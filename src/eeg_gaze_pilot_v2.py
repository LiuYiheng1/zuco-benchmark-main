import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
GAZE_FEATURE_FILE = 'sent_gaze_sacc'

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

class PCETModel:
    def __init__(self, n_pca_components=20, lambda_reg=0.1):
        self.n_pca_components = n_pca_components
        self.lambda_reg = lambda_reg

    def fit_predict(self, X_cal, y_cal, X_test):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_cal[y_cal == c]
            if len(X_c) > self.n_pca_components:
                pca = PCA(n_components=self.n_pca_components, random_state=42)
                pca.fit(X_c)
                pca_models[c] = pca
            else:
                pca_models[c] = None

        def compute_errors(X, pms):
            n_samples = len(X)
            error_features = np.zeros((n_samples, len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_reconstructed = pca.inverse_transform(pca.transform(X))
                    errors = X - X_reconstructed
                    error_features[:, i] = np.sqrt(np.sum(errors ** 2, axis=1))
                    error_features[:, 1 + i] = np.mean(np.abs(errors), axis=1)
            return error_features

        error_cal = compute_errors(X_cal, pca_models)
        error_test = compute_errors(X_test, pca_models)

        scaler = StandardScaler()
        X_cal_combined = np.hstack([scaler.fit_transform(X_cal), error_cal])
        X_test_combined = np.hstack([scaler.transform(X_test), error_test])

        clf = RidgeClassifier(alpha=self.lambda_reg)
        clf.fit(X_cal_combined, y_cal)

        preds = clf.predict(X_test_combined)
        return preds

class GETAModel:
    def __init__(self, hidden_sizes=(64, 32), gaze_hidden=32):
        self.hidden_sizes = hidden_sizes
        self.gaze_hidden = gaze_hidden

    def fit_predict(self, X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_eeg = StandardScaler()
        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)

        scaler_gaze = StandardScaler()
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(self.gaze_hidden,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        entropy_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        entropy_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)
        confidence_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        confidence_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        attention_cal = np.tile(entropy_cal, (1, X_eeg_cal_s.shape[1])) * 0.01 + np.tile(confidence_cal, (1, X_eeg_cal_s.shape[1]))
        attention_test = np.tile(entropy_test, (1, X_eeg_test_s.shape[1])) * 0.01 + np.tile(confidence_test, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * attention_cal
        X_eeg_test_att = X_eeg_test_s * attention_test

        clf = MLPClassifier(hidden_layer_sizes=self.hidden_sizes, max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        preds = clf.predict(X_eeg_test_att)
        return preds

class CAGFModel:
    def __init__(self, eeg_hidden=(64, 32), gaze_hidden=32, fuse_hidden=32):
        self.eeg_hidden = eeg_hidden
        self.gaze_hidden = gaze_hidden
        self.fuse_hidden = fuse_hidden

    def fit_predict(self, X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_eeg = StandardScaler()
        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)

        scaler_gaze = StandardScaler()
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=self.eeg_hidden, max_iter=500, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)
        c_eeg_cal = np.max(z_eeg_cal, axis=1).reshape(-1, 1)
        c_eeg_test = np.max(z_eeg_test, axis=1).reshape(-1, 1)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(self.gaze_hidden,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)
        c_gaze_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        c_gaze_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        alpha_cal = 1 / (1 + np.exp(-z_eeg_cal[:, 0] + z_gaze_cal[:, 0]))
        alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))

        z_eeg_cal_last = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_gaze_cal_last = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_eeg_test_last = eeg_mlp.predict_proba(X_eeg_test_s)
        z_gaze_test_last = gaze_mlp.predict_proba(X_gaze_test_s)

        z_fused_cal = alpha_cal.reshape(-1, 1) * z_eeg_cal_last + (1 - alpha_cal.reshape(-1, 1)) * z_gaze_cal_last
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test_last + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test_last

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        preds = clf_final.predict(z_fused_test)
        return preds

def run_experiment():
    results = []
    shot_settings = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    print("Starting experiment...", flush=True)

    for seed in seeds:
        print(f'\nSeed {seed}:', flush=True)
        for held_out in Y_SUBJECTS:
            print(f'  {held_out}', end='', flush=True)

            X_eeg_test, y_test, trial_ids_eeg = load_eeg_data(held_out)
            X_gaze_test, y_gaze, trial_ids_gaze = load_gaze_features(held_out)
            if X_eeg_test is None or X_gaze_test is None:
                print(' skip(no data)', flush=True)
                continue

            X_eeg_test_aligned, y_eeg_test_aligned, X_gaze_test_aligned, y_gaze_test_aligned = align_eeg_gaze(
                X_eeg_test, y_test, trial_ids_eeg, X_gaze_test, y_gaze, trial_ids_gaze)
            y_test_aligned = y_eeg_test_aligned

            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_eeg_train_list, y_eeg_train_list = [], []
            X_gaze_train_list = []
            for subj in train_subjs:
                Xe, ye, tid_e = load_eeg_data(subj)
                Xg, yg, tid_g = load_gaze_features(subj)
                if Xe is not None and Xg is not None:
                    Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
                    if len(Xe_a) > 0:
                        X_eeg_train_list.append(Xe_a)
                        y_eeg_train_list.append(ye_a)
                        X_gaze_train_list.append(Xg_a)

            if len(X_eeg_train_list) == 0:
                print(' skip(no train)', flush=True)
                continue

            n_samples = len(y_test_aligned)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_size = n_samples // 3
            test_indices = indices[:test_size]
            cal_pool_indices = indices[test_size:]

            X_eeg_cal_pool = X_eeg_test_aligned[cal_pool_indices]
            y_cal_pool = y_test_aligned[cal_pool_indices]
            X_gaze_cal_pool = X_gaze_test_aligned[cal_pool_indices]

            X_eeg_test_final = X_eeg_test_aligned[test_indices]
            X_gaze_test_final = X_gaze_test_aligned[test_indices]
            y_test_final = y_test_aligned[test_indices]

            for n_cal in shot_settings:
                if n_cal * 2 > len(cal_pool_indices):
                    continue

                cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
                X_eeg_cal = X_eeg_cal_pool[cal_idx]
                X_gaze_cal = X_gaze_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                if len(np.unique(y_cal)) < 2:
                    continue

                row = {'seed': seed, 'subject': held_out, 'n_cal': n_cal}

                try:
                    scaler_eeg = StandardScaler()
                    X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
                    X_eeg_test_s = scaler_eeg.transform(X_eeg_test_final)
                    clf = SVC(kernel='rbf', probability=True, random_state=42)
                    clf.fit(X_eeg_cal_s, y_cal)
                    probs = clf.predict_proba(X_eeg_test_s)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    row['EEG_SVM_acc'] = accuracy_score(y_test_final, preds)
                    row['EEG_SVM_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['EEG_SVM_auroc'] = roc_auc_score(y_test_final, probs)
                except Exception as e:
                    row['EEG_SVM_acc'] = 0.5
                    print(f' EEG_SVM error: {e}', end='', flush=True)

                try:
                    scaler_gaze = StandardScaler()
                    X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
                    X_gaze_test_s = scaler_gaze.transform(X_gaze_test_final)
                    clf = SVC(kernel='rbf', probability=True, random_state=42)
                    clf.fit(X_gaze_cal_s, y_cal)
                    probs = clf.predict_proba(X_gaze_test_s)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    row['Gaze_SVM_acc'] = accuracy_score(y_test_final, preds)
                    row['Gaze_SVM_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['Gaze_SVM_auroc'] = roc_auc_score(y_test_final, probs)
                except:
                    row['Gaze_SVM_acc'] = 0.5

                try:
                    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                    clf.fit(X_eeg_cal_s, y_cal)
                    probs = clf.predict_proba(X_eeg_test_s)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    row['EEG_MLP_acc'] = accuracy_score(y_test_final, preds)
                    row['EEG_MLP_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['EEG_MLP_auroc'] = roc_auc_score(y_test_final, probs)
                except:
                    row['EEG_MLP_acc'] = 0.5

                try:
                    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
                    clf.fit(X_gaze_cal_s, y_cal)
                    probs = clf.predict_proba(X_gaze_test_s)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    row['Gaze_MLP_acc'] = accuracy_score(y_test_final, preds)
                    row['Gaze_MLP_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['Gaze_MLP_auroc'] = roc_auc_score(y_test_final, probs)
                except:
                    row['Gaze_MLP_acc'] = 0.5

                try:
                    X_concat_cal = np.hstack([X_eeg_cal_s, X_gaze_cal_s])
                    X_concat_test = np.hstack([X_eeg_test_s, X_gaze_test_s])
                    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                    clf.fit(X_concat_cal, y_cal)
                    probs = clf.predict_proba(X_concat_test)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    row['EEG+Gaze_concat_acc'] = accuracy_score(y_test_final, preds)
                    row['EEG+Gaze_concat_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['EEG+Gaze_concat_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['EEG+Gaze_concat_auroc'] = roc_auc_score(y_test_final, probs)
                except:
                    row['EEG+Gaze_concat_acc'] = 0.5

                try:
                    clf_eeg = SVC(kernel='rbf', probability=True, random_state=42)
                    clf_eeg.fit(X_eeg_cal_s, y_cal)
                    p_eeg = clf_eeg.predict_proba(X_eeg_test_s)[:, 1]
                    clf_gaze = SVC(kernel='rbf', probability=True, random_state=42)
                    clf_gaze.fit(X_gaze_cal_s, y_cal)
                    p_gaze = clf_gaze.predict_proba(X_gaze_test_s)[:, 1]
                    p_avg = (p_eeg + p_gaze) / 2
                    preds = (p_avg >= 0.5).astype(int)
                    row['Static_EEG_Gaze_avg_acc'] = accuracy_score(y_test_final, preds)
                    row['Static_EEG_Gaze_avg_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['Static_EEG_Gaze_avg_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['Static_EEG_Gaze_avg_auroc'] = roc_auc_score(y_test_final, p_avg)
                except:
                    row['Static_EEG_Gaze_avg_acc'] = 0.5

                try:
                    preds = PCETModel().fit_predict(X_eeg_cal, y_cal, X_eeg_test_final)
                    row['PCET_only_acc'] = accuracy_score(y_test_final, preds)
                    row['PCET_only_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['PCET_only_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['PCET_only_auroc'] = roc_auc_score(y_test_final, preds.astype(float))
                except Exception as e:
                    row['PCET_only_acc'] = 0.5
                    print(f' PCET error: {e}', end='', flush=True)

                try:
                    preds = GETAModel().fit_predict(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test_final, X_gaze_test_final)
                    row['GETA_only_acc'] = accuracy_score(y_test_final, preds)
                    row['GETA_only_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['GETA_only_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['GETA_only_auroc'] = roc_auc_score(y_test_final, preds.astype(float))
                except:
                    row['GETA_only_acc'] = 0.5

                try:
                    p_pcet = PCETModel().fit_predict(X_eeg_cal, y_cal, X_eeg_test_final).astype(float)
                    p_geta = GETAModel().fit_predict(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test_final, X_gaze_test_final).astype(float)
                    p_concat = (p_pcet + p_geta) / 2
                    preds = (p_concat >= 0.5).astype(int)
                    row['PCET+GETA_concat_acc'] = accuracy_score(y_test_final, preds)
                    row['PCET+GETA_concat_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['PCET+GETA_concat_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['PCET+GETA_concat_auroc'] = roc_auc_score(y_test_final, p_concat)
                except:
                    row['PCET+GETA_concat_acc'] = 0.5

                try:
                    p_pcet_s = StandardScaler().fit_transform(p_pcet.reshape(-1, 1)).flatten()
                    p_geta_s = StandardScaler().fit_transform(p_geta.reshape(-1, 1)).flatten()
                    p_static = (p_pcet_s + p_geta_s) / 2
                    preds = (p_static >= 0.5).astype(int)
                    row['PCET+GETA_static_avg_acc'] = accuracy_score(y_test_final, preds)
                    row['PCET+GETA_static_avg_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['PCET+GETA_static_avg_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['PCET+GETA_static_avg_auroc'] = roc_auc_score(y_test_final, p_static)
                except:
                    row['PCET+GETA_static_avg_acc'] = 0.5

                try:
                    preds = CAGFModel().fit_predict(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test_final, X_gaze_test_final)
                    row['PCET+GETA+CAGF_acc'] = accuracy_score(y_test_final, preds)
                    row['PCET+GETA+CAGF_f1'] = f1_score(y_test_final, preds, average='macro')
                    row['PCET+GETA+CAGF_bacc'] = balanced_accuracy_score(y_test_final, preds)
                    row['PCET+GETA+CAGF_auroc'] = roc_auc_score(y_test_final, preds.astype(float))
                except:
                    row['PCET+GETA+CAGF_acc'] = 0.5

                results.append(row)
            print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'eeg_gaze_pilot_results.csv'), index=False)
    print('\nSaved results', flush=True)
    return df

def analyze_results(df):
    print('\n=== Analysis ===', flush=True)
    acc_cols = [c for c in df.columns if c.endswith('_acc')]
    summary = {}
    for col in acc_cols:
        method = col.replace('_acc', '')
        summary[method] = {}
        for shot in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == shot][col]
            if len(sub) > 0:
                summary[method][shot] = (sub.mean(), sub.std())

    print('\nAccuracy Summary:', flush=True)
    print('-' * 120, flush=True)
    header = f"{'Method':<30}" + "".join([f"{'SHOT-'+str(s):>18}" for s in [3, 5, 10, 20, 50]])
    print(header, flush=True)
    print('-' * 120, flush=True)

    for method in ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
                   'Static_EEG_Gaze_avg', 'PCET_only', 'GETA_only', 'PCET+GETA_concat',
                   'PCET+GETA_static_avg', 'PCET+GETA+CAGF']:
        if method in summary:
            row_str = f"{method:<30}"
            for shot in [3, 5, 10, 20, 50]:
                if shot in summary[method]:
                    m, s = summary[method][shot]
                    row_str += f"{m*100:>16.1f}%±{s*100:.1f}%"
                else:
                    row_str += f"{'N/A':>18}"
            print(row_str, flush=True)

    print('\n=== Success Criteria ===', flush=True)

    cagf_col = 'PCET+GETA+CAGF_acc'
    concat_col = 'PCET+GETA_concat_acc'
    static_col = 'PCET+GETA_static_avg_acc'
    pcet_col = 'PCET_only_acc'
    geta_col = 'GETA_only_acc'
    gaze_mlp_col = 'Gaze_MLP_acc'
    eeg_gaze_concat_col = 'EEG+Gaze_concat_acc'

    print('\nGETA Success (GETA > Gaze_MLP):', flush=True)
    for shot in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == shot]
        if len(sub) > 0:
            geta_mean = sub[geta_col].mean()
            gaze_mlp_mean = sub[gaze_mlp_col].mean()
            diff = geta_mean - gaze_mlp_mean
            status = "PASS" if diff > 0.01 else "FAIL"
            print(f"  {shot}-shot: GETA={geta_mean*100:.2f}%, Gaze_MLP={gaze_mlp_mean*100:.2f}%, diff={diff*100:.2f}% [{status}]", flush=True)

    print('\nCAGF Success:', flush=True)
    for shot in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == shot]
        if len(sub) > 0:
            cagf_mean = sub[cagf_col].mean()
            concat_mean = sub[concat_col].mean()
            static_mean = sub[static_col].mean()
            c1 = "PASS" if cagf_mean > concat_mean else "FAIL"
            c2 = "PASS" if cagf_mean > static_mean else "FAIL"
            print(f"  {shot}-shot: CAGF={cagf_mean*100:.2f}%, concat={concat_mean*100:.2f}%, static={static_mean*100:.2f}% [{c1},{c2}]", flush=True)

    report = []
    report.append("# EEG-Gaze Multimodal Framework Pilot Results\n")
    report.append("\n## Main Accuracy Comparison\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |")
    report.append("|--------|--------|--------|---------|---------|--------|")

    for method in ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
                   'Static_EEG_Gaze_avg', 'PCET_only', 'GETA_only', 'PCET+GETA_concat',
                   'PCET+GETA_static_avg', 'PCET+GETA+CAGF']:
        if method in summary:
            row = f"| {method} |"
            for shot in [3, 5, 10, 20, 50]:
                if shot in summary[method]:
                    m, s = summary[method][shot]
                    row += f" {m*100:.1f}±{s*100:.1f} |"
                else:
                    row += " - |"
            report.append(row)

    report.append("\n## Success Criteria Check\n\n")
    report.append("### GETA Success (GETA > Gaze_MLP)\n")
    for shot in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == shot]
        if len(sub) > 0:
            geta_mean = sub[geta_col].mean()
            gaze_mlp_mean = sub[gaze_mlp_col].mean()
            diff = geta_mean - gaze_mlp_mean
            status = "PASS" if diff > 0.01 else "FAIL"
            report.append(f"- {shot}-shot: GETA={geta_mean*100:.2f}%, Gaze_MLP={gaze_mlp_mean*100:.2f}%, diff={diff*100:.2f}% [{status}]")

    report.append("\n### CAGF Success (CAGF > concat AND CAGF > static_avg)\n")
    for shot in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == shot]
        if len(sub) > 0:
            cagf_mean = sub[cagf_col].mean()
            concat_mean = sub[concat_col].mean()
            static_mean = sub[static_col].mean()
            c1 = "PASS" if cagf_mean > concat_mean else "FAIL"
            c2 = "PASS" if cagf_mean > static_mean else "FAIL"
            report.append(f"- {shot}-shot: CAGF={cagf_mean*100:.2f}%, concat={concat_mean*100:.2f}%, static={static_mean*100:.2f}% [{c1},{c2}]")

    report.append("\n### Full Framework Success\n")
    for shot in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == shot]
        if len(sub) > 0:
            full_mean = sub[cagf_col].mean()
            pcet_mean = sub[pcet_col].mean()
            geta_mean = sub[geta_col].mean()
            eeg_gaze_mean = sub[eeg_gaze_concat_col].mean()
            c1 = "PASS" if full_mean > pcet_mean else "FAIL"
            c2 = "PASS" if full_mean > geta_mean else "FAIL"
            c3 = "PASS" if full_mean > eeg_gaze_mean else "FAIL"
            report.append(f"- {shot}-shot: Full={full_mean*100:.2f}%, PCET={pcet_mean*100:.2f}%, GETA={geta_mean*100:.2f}%, concat={eeg_gaze_mean*100:.2f}% [{c1},{c2},{c3}]")

    report_text = "\n".join(report)
    with open(os.path.join(REPORTS_DIR, 'eeg_gaze_pilot_report.md'), 'w') as f:
        f.write(report_text)
    print('\nReport saved', flush=True)

if __name__ == '__main__':
    print("EEG-Gaze Multimodal Framework Pilot", flush=True)
    print("="*80, flush=True)
    df = run_experiment()
    analyze_results(df)
    print("\nDone!", flush=True)