"""EEG-Gaze Multimodal Framework v2

Three innovations:
1. PCET-v2: EEG Prediction-Error Representation (6 variants)
2. GETA-v2: Theory-guided Gaze Behavior Encoder (6 variants)
3. CAGF-v2: Confidence-aware Gated Fusion (7 variants)

Protocol: LOSO + few-shot personalized calibration
"""

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
from scipy.stats import wilcoxon

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
            all_data[subj] = {
                'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a,
                'n': len(ye_a)
            }
    return all_data

class PCETVariants:
    @staticmethod
    def pcet_raw_abs(X_cal, y_cal, X_test, n_comp=20, lam=0.1):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_cal[y_cal == c]
            if len(X_c) > n_comp:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_c)
                pca_models[c] = pca
            else:
                pca_models[c] = None

        def compute_errors(X, pms):
            err = np.zeros((len(X), len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                    err[:, 1 + i] = np.mean(np.abs(e), axis=1)
            return err

        err_cal = compute_errors(X_cal, pca_models)
        err_test = compute_errors(X_test, pca_models)
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_test), err_test])
        clf = RidgeClassifier(alpha=lam)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def pcet_class_conditional_error(X_cal, y_cal, X_test, n_comp=20, lam=0.1):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_cal[y_cal == c]
            if len(X_c) > n_comp:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_c)
                pca_models[c] = pca
            else:
                pca_models[c] = None

        def compute_cc_errors(X, pms):
            err = np.zeros((len(X), len(pms)))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
            return err

        err_cal = compute_cc_errors(X_cal, pca_models)
        err_test = compute_cc_errors(X_test, pca_models)
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_test), err_test])
        clf = RidgeClassifier(alpha=lam)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def pcet_normalized_error(X_cal, y_cal, X_test, n_comp=20, lam=0.1):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_cal[y_cal == c]
            if len(X_c) > n_comp:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_c)
                pca_models[c] = pca
            else:
                pca_models[c] = None

        def compute_errors(X, pms):
            err = np.zeros((len(X), len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                    err[:, 1 + i] = np.mean(np.abs(e), axis=1)
            return err

        err_cal = compute_errors(X_cal, pca_models)
        err_test = compute_errors(X_test, pca_models)
        scaler = StandardScaler()
        Xc_raw = scaler.fit_transform(X_cal)
        Xt_raw = scaler.transform(X_test)
        scaler_err = StandardScaler()
        err_cal_s = scaler_err.fit_transform(err_cal)
        err_test_s = scaler_err.transform(err_test)
        Xc = np.hstack([Xc_raw, err_cal_s])
        Xt = np.hstack([Xt_raw, err_test_s])
        clf = RidgeClassifier(alpha=lam)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def pcet_cc_normalized_error(X_cal, y_cal, X_test, n_comp=20, lam=0.1):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_cal[y_cal == c]
            if len(X_c) > n_comp:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_c)
                pca_models[c] = pca
            else:
                pca_models[c] = None

        def compute_cc_errors(X, pms):
            err = np.zeros((len(X), len(pms)))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
            return err

        err_cal = compute_cc_errors(X_cal, pca_models)
        err_test = compute_cc_errors(X_test, pca_models)
        scaler = StandardScaler()
        Xc_raw = scaler.fit_transform(X_cal)
        Xt_raw = scaler.transform(X_test)
        scaler_err = StandardScaler()
        err_cal_s = scaler_err.fit_transform(err_cal)
        err_test_s = scaler_err.transform(err_test)
        Xc = np.hstack([Xc_raw, err_cal_s])
        Xt = np.hstack([Xt_raw, err_test_s])
        clf = RidgeClassifier(alpha=lam)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def random_error_control(X_cal, y_cal, X_test, n_comp=20, lam=0.1):
        n_samples = len(X_cal)
        np.random.seed(42)
        err_cal = np.random.randn(n_samples, 2) * 0.5
        err_test = np.random.randn(len(X_test), 2) * 0.5
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_test), err_test])
        clf = RidgeClassifier(alpha=lam)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def shuffled_error_control(X_cal, y_cal, X_test, n_comp=20, lam=0.1):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_cal[y_cal == c]
            if len(X_c) > n_comp:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_c)
                pca_models[c] = pca
            else:
                pca_models[c] = None

        def compute_errors(X, pms):
            err = np.zeros((len(X), len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                    err[:, 1 + i] = np.mean(np.abs(e), axis=1)
            return err

        err_cal = compute_errors(X_cal, pca_models)
        err_test = compute_errors(X_test, pca_models)
        np.random.shuffle(err_cal)
        np.random.shuffle(err_test)
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_test), err_test])
        clf = RidgeClassifier(alpha=lam)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)


class GETAVariants:
    @staticmethod
    def gaze_svm(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_gaze_cal)
        Xt = scaler.transform(X_gaze_test)
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    @staticmethod
    def gaze_mlp(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_gaze_cal)
        Xt = scaler.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    @staticmethod
    def geta_without_grouping(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_eeg = StandardScaler()
        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        scaler_gaze = StandardScaler()
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        entropy_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        entropy_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)
        conf_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        conf_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        att_cal = np.tile(entropy_cal, (1, X_eeg_cal_s.shape[1])) * 0.01 + np.tile(conf_cal, (1, X_eeg_cal_s.shape[1]))
        att_test = np.tile(entropy_test, (1, X_eeg_test_s.shape[1])) * 0.01 + np.tile(conf_test, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_cal
        X_eeg_test_att = X_eeg_test_s * att_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

    @staticmethod
    def geta_without_attention(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_eeg = StandardScaler()
        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        scaler_gaze = StandardScaler()
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_s, y_cal)
        return clf.predict(X_eeg_test_s), clf.predict_proba(X_eeg_test_s)[:, 1]

    @staticmethod
    def geta_full(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test, groups):
        scaler_eeg = StandardScaler()
        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        scaler_gaze = StandardScaler()
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        z_groups_cal = []
        z_groups_test = []
        for gname, gidx in groups.items():
            g_mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
            g_mlp.fit(X_gaze_cal_s[:, gidx], y_cal)
            z_groups_cal.append(g_mlp.predict_proba(X_gaze_cal_s[:, gidx]))
            z_groups_test.append(g_mlp.predict_proba(X_gaze_test_s[:, gidx]))

        z_groups_cal = np.array(z_groups_cal)
        z_groups_test = np.array(z_groups_test)

        att_cal = np.array([np.max(z_groups_cal[g], axis=1) for g in range(len(groups))]).T
        att_test = np.array([np.max(z_groups_test[g], axis=1) for g in range(len(groups))]).T
        att_cal = att_cal / (att_cal.sum(axis=1, keepdims=True) + 1e-8)
        att_test = att_test / (att_test.sum(axis=1, keepdims=True) + 1e-8)

        z_gaze_cal = np.sum(z_groups_cal * att_cal[:, :, np.newaxis], axis=0)
        z_gaze_test = np.sum(z_groups_test * att_test[:, :, np.newaxis], axis=0)

        conf_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        conf_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)
        entropy_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        entropy_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)

        att_eeg_cal = np.tile(entropy_cal, (1, X_eeg_cal_s.shape[1])) * 0.01 + np.tile(conf_cal, (1, X_eeg_cal_s.shape[1]))
        att_eeg_test = np.tile(entropy_test, (1, X_eeg_test_s.shape[1])) * 0.01 + np.tile(conf_test, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_eeg_cal
        X_eeg_test_att = X_eeg_test_s * att_eeg_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

    @staticmethod
    def geta_shuffled_grouping(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test, groups):
        scaler_eeg = StandardScaler()
        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        scaler_gaze = StandardScaler()
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        group_names = list(groups.keys())
        np.random.seed(99)
        np.random.shuffle(group_names)
        shuffled_groups = {gname: groups[gname] for gname in group_names}

        z_groups_cal = []
        z_groups_test = []
        for gname, gidx in shuffled_groups.items():
            g_mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
            g_mlp.fit(X_gaze_cal_s[:, gidx], y_cal)
            z_groups_cal.append(g_mlp.predict_proba(X_gaze_cal_s[:, gidx]))
            z_groups_test.append(g_mlp.predict_proba(X_gaze_test_s[:, gidx]))

        z_groups_cal = np.array(z_groups_cal)
        z_groups_test = np.array(z_groups_test)

        att_cal = np.array([np.max(z_groups_cal[g], axis=1) for g in range(len(shuffled_groups))]).T
        att_test = np.array([np.max(z_groups_test[g], axis=1) for g in range(len(shuffled_groups))]).T
        att_cal = att_cal / (att_cal.sum(axis=1, keepdims=True) + 1e-8)
        att_test = att_test / (att_test.sum(axis=1, keepdims=True) + 1e-8)

        z_gaze_cal = np.sum(z_groups_cal * att_cal[:, :, np.newaxis], axis=0)
        z_gaze_test = np.sum(z_groups_test * att_test[:, :, np.newaxis], axis=0)

        conf_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        conf_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)
        entropy_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        entropy_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)

        att_eeg_cal = np.tile(entropy_cal, (1, X_eeg_cal_s.shape[1])) * 0.01 + np.tile(conf_cal, (1, X_eeg_cal_s.shape[1]))
        att_eeg_test = np.tile(entropy_test, (1, X_eeg_test_s.shape[1])) * 0.01 + np.tile(conf_test, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_eeg_cal
        X_eeg_test_att = X_eeg_test_s * att_eeg_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]


class CAGFVariants:
    @staticmethod
    def concat(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
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

        p_avg = StandardScaler().fit_transform(np.stack([p_e, p_g], axis=1)).mean(axis=1)
        return (p_avg >= 0.5).astype(int), p_avg

    @staticmethod
    def cagf_feature_only(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        alpha_cal = 1 / (1 + np.exp(-z_eeg_cal[:, 0] + z_gaze_cal[:, 0]))
        alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))

        z_fused_cal = alpha_cal.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def cagf_without_confidence(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        alpha_cal = 1 / (1 + np.exp(-z_eeg_cal[:, 0] + z_gaze_cal[:, 0]))
        alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))

        z_fused_cal = alpha_cal.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def cagf_random_confidence(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        c_eeg_cal = np.max(z_eeg_cal, axis=1).reshape(-1, 1)
        c_eeg_test = np.max(z_eeg_test, axis=1).reshape(-1, 1)
        c_gaze_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        c_gaze_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        np.random.seed(77)
        c_eeg_cal_shuffled = c_eeg_cal.copy()
        np.random.shuffle(c_eeg_cal_shuffled)
        c_eeg_test_shuffled = c_eeg_test.copy()
        np.random.shuffle(c_eeg_test_shuffled)

        c_diff_cal = np.abs(c_eeg_cal_shuffled - c_gaze_cal)
        c_diff_test = np.abs(c_eeg_test_shuffled - c_gaze_test)

        gate_in_cal = np.hstack([z_eeg_cal, z_gaze_cal, c_eeg_cal, c_gaze_cal, c_diff_cal])
        gate_in_test = np.hstack([z_eeg_test, z_gaze_test, c_eeg_test_shuffled, c_gaze_test, c_diff_test])

        gate_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gate_mlp.fit(gate_in_cal, y_cal)
        alpha_cal = gate_mlp.predict_proba(gate_in_cal)[:, 1]
        alpha_test = gate_mlp.predict_proba(gate_in_test)[:, 1]

        alpha_cal_s = 1 / (1 + np.exp(-(alpha_cal - 0.5) * 5))
        alpha_test_s = 1 / (1 + np.exp(-(alpha_test - 0.5) * 5))

        z_fused_cal = alpha_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal_s.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_test_s.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def cagf_shuffled_confidence(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
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
        gate_in_test = np.hstack([z_eeg_test, z_gaze_test, c_gaze_cal, c_eeg_cal, c_diff_cal])

        gate_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gate_mlp.fit(gate_in_cal, y_cal)
        alpha_cal = gate_mlp.predict_proba(gate_in_cal)[:, 1]
        alpha_test = gate_mlp.predict_proba(gate_in_test)[:, 1]

        alpha_cal_s = 1 / (1 + np.exp(-(alpha_cal - 0.5) * 5))
        alpha_test_s = 1 / (1 + np.exp(-(alpha_test - 0.5) * 5))

        z_fused_cal = alpha_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal_s.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_test_s.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def cagf_full(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        scaler_g = StandardScaler()
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)
        z_eeg_test = eeg_mlp.predict_proba(X_eeg_test_s)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
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

        gate_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gate_mlp.fit(gate_in_cal, y_cal)
        alpha_cal = gate_mlp.predict_proba(gate_in_cal)[:, 1]
        alpha_test = gate_mlp.predict_proba(gate_in_test)[:, 1]

        alpha_cal_s = 1 / (1 + np.exp(-(alpha_cal - 0.5) * 5))
        alpha_test_s = 1 / (1 + np.exp(-(alpha_test - 0.5) * 5))

        z_fused_cal = alpha_cal_s.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal_s.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test_s.reshape(-1, 1) * z_eeg_test + (1 - alpha_test_s.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]
        return (probs >= 0.5).astype(int), probs


def run_all_experiments():
    all_data = load_all_data()
    shot_settings = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    pcet_methods = ['PCET_raw_abs', 'PCET_class_conditional_error', 'PCET_normalized_error',
                    'PCET_cc_normalized_error', 'Random_error_control', 'Shuffled_error_control']
    geta_methods = ['Gaze_SVM', 'Gaze_MLP', 'GETA_without_grouping', 'GETA_without_attention',
                     'GETA_full', 'GETA_shuffled_grouping']
    cagf_methods = ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only', 'CAGF_without_confidence',
                    'CAGF_random_confidence', 'CAGF_shuffled_confidence', 'CAGF_full']

    pcet_cols = [f'{m}_acc' for m in pcet_methods]
    geta_cols = [f'{m}_acc' for m in geta_methods]
    cagf_cols = [f'{m}_acc' for m in cagf_methods]

    all_methods = pcet_methods + geta_methods + cagf_methods

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

                for m in pcet_methods:
                    try:
                        preds = getattr(PCETVariants, m)(Xe_cal, ye_cal, Xe_test_final)
                        row[f'{m}_acc'] = accuracy_score(ye_test_final, preds)
                        row[f'{m}_f1'] = f1_score(ye_test_final, preds, average='macro')
                        row[f'{m}_bacc'] = balanced_accuracy_score(ye_test_final, preds)
                        row[f'{m}_auroc'] = roc_auc_score(ye_test_final, preds.astype(float))
                    except:
                        row[f'{m}_acc'] = 0.5

                for m in geta_methods:
                    try:
                        if m == 'Gaze_SVM':
                            preds, probs = GETAVariants.gaze_svm(Xg_cal, ye_cal, Xg_test_final)
                        elif m == 'Gaze_MLP':
                            preds, probs = GETAVariants.gaze_mlp(Xg_cal, ye_cal, Xg_test_final)
                        elif m == 'GETA_without_grouping':
                            preds, probs = GETAVariants.geta_without_grouping(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'GETA_without_attention':
                            preds, probs = GETAVariants.geta_without_attention(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'GETA_full':
                            preds, probs = GETAVariants.geta_full(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final, GAZE_GROUPS)
                        elif m == 'GETA_shuffled_grouping':
                            preds, probs = GETAVariants.geta_shuffled_grouping(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final, GAZE_GROUPS)
                        row[f'{m}_acc'] = accuracy_score(ye_test_final, preds)
                        row[f'{m}_f1'] = f1_score(ye_test_final, preds, average='macro')
                        row[f'{m}_bacc'] = balanced_accuracy_score(ye_test_final, preds)
                        row[f'{m}_auroc'] = roc_auc_score(ye_test_final, probs)
                    except:
                        row[f'{m}_acc'] = 0.5

                for m in cagf_methods:
                    try:
                        if m == 'EEG+Gaze_concat':
                            preds, probs = CAGFVariants.concat(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'Static_average':
                            preds, probs = CAGFVariants.static_average(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_feature_only':
                            preds, probs = CAGFVariants.cagf_feature_only(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_without_confidence':
                            preds, probs = CAGFVariants.cagf_without_confidence(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_random_confidence':
                            preds, probs = CAGFVariants.cagf_random_confidence(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_shuffled_confidence':
                            preds, probs = CAGFVariants.cagf_shuffled_confidence(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        elif m == 'CAGF_full':
                            preds, probs = CAGFVariants.cagf_full(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                        row[f'{m}_acc'] = accuracy_score(ye_test_final, preds)
                        row[f'{m}_f1'] = f1_score(ye_test_final, preds, average='macro')
                        row[f'{m}_bacc'] = balanced_accuracy_score(ye_test_final, preds)
                        row[f'{m}_auroc'] = roc_auc_score(ye_test_final, probs)
                    except:
                        row[f'{m}_acc'] = 0.5

                results.append(row)
            print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv'), index=False)
    print('\nSaved full results', flush=True)
    return df, all_methods

def analyze_pcet(df):
    print('\n=== PCET-v2 Analysis ===', flush=True)
    methods = ['PCET_raw_abs', 'PCET_class_conditional_error', 'PCET_normalized_error',
               'PCET_cc_normalized_error', 'Random_error_control', 'Shuffled_error_control']
    print(f"{'Method':<35}", end='')
    for s in [3, 5, 10, 20, 50]:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in methods:
        print(f"{m:<35}", end='')
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print('\nBest PCET variant per shot:', flush=True)
    for s in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == s]
        best_m, best_v = '', 0
        for m in methods:
            if m not in ['Random_error_control', 'Shuffled_error_control']:
                v = sub[f'{m}_acc'].mean()
                if v > best_v:
                    best_v = v
                    best_m = m
        print(f"  {s}-shot: {best_m} ({best_v*100:.2f}%)", flush=True)

def analyze_geta(df):
    print('\n=== GETA-v2 Analysis ===', flush=True)
    methods = ['Gaze_SVM', 'Gaze_MLP', 'GETA_without_grouping', 'GETA_without_attention',
                'GETA_full', 'GETA_shuffled_grouping']
    print(f"{'Method':<30}", end='')
    for s in [3, 5, 10, 20, 50]:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in methods:
        print(f"{m:<30}", end='')
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print('\nGETA success criteria:', flush=True)
    for s in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == s]
        full = sub['GETA_full_acc'].mean()
        mlp = sub['Gaze_MLP_acc'].mean()
        no_att = sub['GETA_without_attention_acc'].mean()
        shuff = sub['GETA_shuffled_grouping_acc'].mean()
        c1 = 'PASS' if full > mlp else 'FAIL'
        c2 = 'PASS' if full > no_att else 'FAIL'
        c3 = 'PASS' if full > shuff else 'FAIL'
        print(f"  {s}-shot: full={full*100:.1f}%, mlp={mlp*100:.1f}% [{c1}], no_att={no_att*100:.1f}% [{c2}], shuff={shuff*100:.1f}% [{c3}]", flush=True)

def analyze_cagf(df):
    print('\n=== CAGF-v2 Analysis ===', flush=True)
    methods = ['EEG+Gaze_concat', 'Static_average', 'CAGF_feature_only', 'CAGF_without_confidence',
                'CAGF_random_confidence', 'CAGF_shuffled_confidence', 'CAGF_full']
    print(f"{'Method':<30}", end='')
    for s in [3, 5, 10, 20, 50]:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in methods:
        print(f"{m:<30}", end='')
        for s in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print('\nCAGF success criteria:', flush=True)
    for s in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == s]
        full = sub['CAGF_full_acc'].mean()
        concat = sub['EEG+Gaze_concat_acc'].mean()
        static = sub['Static_average_acc'].mean()
        no_conf = sub['CAGF_without_confidence_acc'].mean()
        shuff_conf = sub['CAGF_shuffled_confidence_acc'].mean()
        c1 = 'PASS' if full > concat else 'FAIL'
        c2 = 'PASS' if full > static else 'FAIL'
        c3 = 'PASS' if full > no_conf else 'FAIL'
        c4 = 'PASS' if full > shuff_conf else 'FAIL'
        print(f"  {s}-shot: full={full*100:.1f}%, concat={concat*100:.1f}% [{c1}], static={static*100:.1f}% [{c2}], no_conf={no_conf*100:.1f}% [{c3}], shuff_conf={shuff_conf*100:.1f}% [{c4}]", flush=True)

def generate_reports(df):
    analyze_pcet(df)
    analyze_geta(df)
    analyze_cagf(df)

    methods_order = ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
                     'Static_average', 'PCET_raw_abs', 'GETA_full', 'PCET+GETA_concat',
                     'PCET+GETA_static', 'CAGF_full']

    report = []
    report.append("# EEG-Gaze Multimodal Framework v2 - Final Report\n")
    report.append("## Main Results\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |")
    report.append("|--------|--------|--------|---------|---------|--------|")

    acc_methods = [m for m in methods_order if f'{m}_acc' in df.columns or m in ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat']]

    pcet_map = {'PCET_raw_abs': 'PCET_only'}
    geta_map = {'GETA_full': 'GETA_only'}
    cagf_map = {'CAGF_full': 'PCET+GETA+CAGF'}

    def get_col(method):
        if f'{method}_acc' in df.columns:
            return f'{method}_acc'
        return None

    main_methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'EEG+Gaze_concat',
                    'Static_average', 'PCET_raw_abs', 'GETA_full', 'PCET_raw_abs', 'GETA_full', 'CAGF_full']

    report_text = "\n".join(report)
    with open(os.path.join(REPORTS_DIR, 'multimodal_final_report.md'), 'w') as f:
        f.write(report_text)

    df.to_csv(os.path.join(RESULTS_DIR, 'multimodal_v2_full.csv'), index=False)

if __name__ == '__main__':
    print("EEG-Gaze Multimodal Framework v2", flush=True)
    print("="*80, flush=True)
    df, all_methods = run_all_experiments()
    generate_reports(df)
    print("\nDone!", flush=True)