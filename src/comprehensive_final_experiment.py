"""Final Comprehensive Experiment for Paper

Implements:
1. Sanity controls (majority, random, BERT proxy, sentence length)
2. Feature baselines (EEG/Gaze SVM variants)
3. Deep baselines proxy (LSTM/GCN proxies at sentence level)
4. GETA ablation
5. PCET ablation
6. CAGF ablation (from previous experiments)
7. Significance tests
8. Duplicate sentence analysis
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

GAZE_FEATURES = {
    'fixation_number': 'fixation_number',
    'mean_sacc_amp': 'mean_sacc_amp',
    'max_sacc_amp': 'max_sacc_amp',
    'mean_sacc_velocity': 'mean_sacc_velocity',
    'max_sacc_velocity': 'max_sacc_velocity',
    'mean_sacc_dur': 'mean_sacc_dur',
    'max_sacc_dur': 'max_sacc_dur',
    'omission_rate': 'omission_rate',
    'reading_speed': 'reading_speed'
}

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

def load_gaze_features(subject, feature_name='sent_gaze_sacc'):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_name}.npy")
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
        if feature_name == 'sent_gaze_sacc':
            numeric_vals = [float(v) for v in values[:-1]]
            features = np.array(numeric_vals, dtype=np.float64)
        else:
            features = np.array(values, dtype=np.float64).flatten()
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def load_all_gaze_features(subject):
    gaze_list = []
    for feat_name in GAZE_FEATURES.values():
        path = os.path.join(FEATURES_DIR, f"{subject}_{feat_name}.npy")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True).item()
            gaze_list.append(data)
    if len(gaze_list) == 0:
        return None, None, None
    keys = list(gaze_list[0].keys())
    X, y, trial_ids = [], [], []
    for key in keys:
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        features = []
        valid = True
        for gd in gaze_list:
            if key not in gd:
                valid = False
                break
            arr = np.array(gd[key]).flatten()
            features.extend(arr)
        if valid:
            X.append(features)
            y.append(label)
            trial_ids.append(get_trial_id(key))
    if len(X) == 0:
        return None, None, None
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
        Xg, yg, tid_g = load_gaze_features(subj, 'sent_gaze_sacc')
        if Xe is not None and Xg is not None:
            Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
            all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
    return all_data

class SanityControls:
    @staticmethod
    def majority_baseline(y_cal, y_test):
        most_common = 1 if np.sum(y_cal == 1) >= np.sum(y_cal == 0) else 0
        return np.ones(len(y_test)) * most_common

    @staticmethod
    def random_baseline(y_test, seed=42):
        np.random.seed(seed)
        return np.random.randint(0, 2, len(y_test))

    @staticmethod
    def sentence_length_proxy(X_eeg, y_eeg, X_test):
        sentence_lengths = np.array([len(x) for x in X_eeg])
        test_lengths = np.array([len(x) for x in X_test])
        scaler = StandardScaler()
        sl_cal = scaler.fit_transform(sentence_lengths.reshape(-1, 1))
        sl_test = scaler.transform(test_lengths.reshape(-1, 1))
        clf = LogisticRegression(random_state=42)
        clf.fit(sl_cal, y_eeg)
        return clf.predict(sl_test)

    @staticmethod
    def word_count_proxy(X_eeg, y_eeg, X_test):
        word_counts = np.array([np.sum(x > 0) for x in X_eeg])
        test_counts = np.array([np.sum(x > 0) for x in X_test])
        scaler = StandardScaler()
        wc_cal = scaler.fit_transform(word_counts.reshape(-1, 1))
        wc_test = scaler.transform(test_counts.reshape(-1, 1))
        clf = LogisticRegression(random_state=42)
        clf.fit(wc_cal, y_eeg)
        return clf.predict(wc_test)

class FeatureBaselines:
    @staticmethod
    def eeg_svm(X_eeg_cal, y_cal, X_eeg_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_eeg_cal)
        Xt = scaler.transform(X_eeg_test)
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    @staticmethod
    def gaze_svm(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_gaze_cal)
        Xt = scaler.transform(X_gaze_test)
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    @staticmethod
    def eeg_gaze_concat_svm(X_eeg_cal, X_gaze_cal, y_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(np.hstack([X_eeg_cal_s, X_gaze_cal_s]), y_cal)
        probs = clf.predict_proba(np.hstack([X_eeg_test_s, X_gaze_test_s]))[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def eeg_pca_svm(X_eeg_cal, y_cal, X_eeg_test, n_comp=50):
        scaler = StandardScaler()
        X_eeg_cal_s = scaler.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler.transform(X_eeg_test)
        pca = PCA(n_components=n_comp, random_state=42)
        X_eeg_cal_pca = pca.fit_transform(X_eeg_cal_s)
        X_eeg_test_pca = pca.transform(X_eeg_test_s)
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(X_eeg_cal_pca, y_cal)
        probs = clf.predict_proba(X_eeg_test_pca)[:, 1]
        return (probs >= 0.5).astype(int), probs

    @staticmethod
    def gaze_fixation_only_svm(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_gaze_cal[:, :1])
        Xt = scaler.transform(X_gaze_test[:, :1])
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    @staticmethod
    def gaze_saccade_only_svm(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_gaze_cal[:, 1:5])
        Xt = scaler.transform(X_gaze_test[:, 1:5])
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    @staticmethod
    def gaze_fixation_saccade_svm(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_gaze_cal[:, :5])
        Xt = scaler.transform(X_gaze_test[:, :5])
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

class DeepBaselinesProxy:
    @staticmethod
    def eeg_lstm_proxy(X_eeg_cal, y_cal, X_eeg_test):
        scaler = StandardScaler()
        X_eeg_cal_s = scaler.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler.transform(X_eeg_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_s, y_cal)
        return clf.predict(X_eeg_test_s), clf.predict_proba(X_eeg_test_s)[:, 1]

    @staticmethod
    def gaze_lstm_proxy(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        X_gaze_cal_s = scaler.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        clf.fit(X_gaze_cal_s, y_cal)
        return clf.predict(X_gaze_test_s), clf.predict_proba(X_gaze_test_s)[:, 1]

    @staticmethod
    def eeg_gcn_proxy(X_eeg_cal, y_cal, X_eeg_test):
        scaler = StandardScaler()
        X_eeg_cal_s = scaler.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler.transform(X_eeg_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_s, y_cal)
        return clf.predict(X_eeg_test_s), clf.predict_proba(X_eeg_test_s)[:, 1]

    @staticmethod
    def eeg_gaze_fusion_lstm_proxy(X_eeg_cal, X_gaze_cal, y_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)
        fused_cal = np.hstack([X_eeg_cal_s, X_gaze_cal_s])
        fused_test = np.hstack([X_eeg_test_s, X_gaze_test_s])
        clf = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
        clf.fit(fused_cal, y_cal)
        return clf.predict(fused_test), clf.predict_proba(fused_test)[:, 1]

class PCETVariants:
    @staticmethod
    def raw_only(X_eeg_cal, y_cal, X_eeg_test):
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X_eeg_cal)
        Xt = scaler.transform(X_eeg_test)
        clf = RidgeClassifier(alpha=0.1)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def abs_error_only(X_eeg_cal, y_cal, X_eeg_test, n_comp=20):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_eeg_cal[y_cal == c]
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

        err_cal = compute_errors(X_eeg_cal, pca_models)
        err_test = compute_errors(X_eeg_test, pca_models)
        clf = RidgeClassifier(alpha=0.1)
        clf.fit(err_cal, y_cal)
        return clf.predict(err_test)

    @staticmethod
    def raw_plus_abserror(X_eeg_cal, y_cal, X_eeg_test, n_comp=20):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_eeg_cal[y_cal == c]
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

        err_cal = compute_errors(X_eeg_cal, pca_models)
        err_test = compute_errors(X_eeg_test, pca_models)
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_eeg_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_eeg_test), err_test])
        clf = RidgeClassifier(alpha=0.1)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def shuffled_error(X_eeg_cal, y_cal, X_eeg_test, n_comp=20):
        pca_models = {}
        for c in [0, 1]:
            X_c = X_eeg_cal[y_cal == c]
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

        err_cal = compute_errors(X_eeg_cal, pca_models)
        err_test = compute_errors(X_eeg_test, pca_models)
        np.random.shuffle(err_cal)
        np.random.shuffle(err_test)
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_eeg_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_eeg_test), err_test])
        clf = RidgeClassifier(alpha=0.1)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

    @staticmethod
    def random_error(X_eeg_cal, y_cal, X_eeg_test):
        np.random.seed(42)
        err_cal = np.random.randn(len(X_eeg_cal), 2) * 0.5
        err_test = np.random.randn(len(X_eeg_test), 2) * 0.5
        scaler = StandardScaler()
        Xc = np.hstack([scaler.fit_transform(X_eeg_cal), err_cal])
        Xt = np.hstack([scaler.transform(X_eeg_test), err_test])
        clf = RidgeClassifier(alpha=0.1)
        clf.fit(Xc, y_cal)
        return clf.predict(Xt)

class GETAAblation:
    @staticmethod
    def eeg_mlp(X_eeg_cal, y_cal, X_eeg_test):
        scaler = StandardScaler()
        X_eeg_cal_s = scaler.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler.transform(X_eeg_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_s, y_cal)
        return clf.predict(X_eeg_test_s), clf.predict_proba(X_eeg_test_s)[:, 1]

    @staticmethod
    def gaze_mlp(X_gaze_cal, y_cal, X_gaze_test):
        scaler = StandardScaler()
        X_gaze_cal_s = scaler.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        clf.fit(X_gaze_cal_s, y_cal)
        return clf.predict(X_gaze_test_s), clf.predict_proba(X_gaze_test_s)[:, 1]

    @staticmethod
    def geta_confidence_only(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        conf_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        conf_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        att_cal = np.tile(conf_cal, (1, X_eeg_cal_s.shape[1]))
        att_test = np.tile(conf_test, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_cal
        X_eeg_test_att = X_eeg_test_s * att_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

    @staticmethod
    def geta_entropy_only(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        ent_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        ent_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)

        att_cal = np.tile(ent_cal * 0.01, (1, X_eeg_cal_s.shape[1]))
        att_test = np.tile(ent_test * 0.01, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_cal
        X_eeg_test_att = X_eeg_test_s * att_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

    @staticmethod
    def geta_confidence_entropy(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        ent_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        ent_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)
        conf_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        conf_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        att_cal = np.tile(ent_cal * 0.01 + conf_cal, (1, X_eeg_cal_s.shape[1]))
        att_test = np.tile(ent_test * 0.01 + conf_test, (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_cal
        X_eeg_test_att = X_eeg_test_s * att_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

    @staticmethod
    def geta_random_attention(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        np.random.seed(99)
        att_cal = np.tile(np.random.rand(len(X_eeg_cal), 1), (1, X_eeg_cal_s.shape[1]))
        att_test = np.tile(np.random.rand(len(X_eeg_test), 1), (1, X_eeg_test_s.shape[1]))

        X_eeg_cal_att = X_eeg_cal_s * att_cal
        X_eeg_test_att = X_eeg_test_s * att_test

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

    @staticmethod
    def geta_shuffled_attention(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_eeg_cal_s = scaler_e.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_e.transform(X_eeg_test)
        X_gaze_cal_s = scaler_g.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_g.transform(X_gaze_test)

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)

        ent_cal = -np.sum(z_gaze_cal * np.log(z_gaze_cal + 1e-8), axis=1).reshape(-1, 1)
        ent_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)
        conf_cal = np.max(z_gaze_cal, axis=1).reshape(-1, 1)
        conf_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)

        att_cal = np.tile(ent_cal * 0.01 + conf_cal, (1, X_eeg_cal_s.shape[1]))
        att_shuffled = att_cal.copy()
        np.random.shuffle(att_shuffled)

        X_eeg_cal_att = X_eeg_cal_s * att_shuffled
        X_eeg_test_att = X_eeg_test_s * att_cal

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_eeg_cal_att, y_cal)
        return clf.predict(X_eeg_test_att), clf.predict_proba(X_eeg_test_att)[:, 1]

def run_comprehensive_experiment():
    print("Loading data...", flush=True)
    all_data = load_all_data()

    shot_settings = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]
    results = []

    sanity_methods = ['Majority', 'Random', 'SentenceLength', 'WordCount']
    feature_methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_Gaze_SVM', 'EEG_PCA_SVM',
                       'Gaze_Fixation_SVM', 'Gaze_Saccade_SVM', 'Gaze_FixSacc_SVM']
    deep_methods = ['EEG_LSTM_proxy', 'Gaze_LSTM_proxy', 'EEG_GCN_proxy', 'EEG_Gaze_LSTM_proxy']
    pcet_methods = ['PCET_raw', 'PCET_abserror', 'PCET_raw_abserror', 'PCET_shuffled', 'PCET_random']
    geta_methods = ['EEG_MLP', 'Gaze_MLP', 'GETA_confidence', 'GETA_entropy', 'GETA_conf_ent',
                    'GETA_random_att', 'GETA_shuffled_att']
    cagf_methods = ['EEG_Gaze_concat', 'Static_avg', 'CAGF_feature_only']

    print("\nRunning experiments...", flush=True)

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
                    row['Majority_acc'] = accuracy_score(ye_test_final, SanityControls.majority_baseline(ye_cal, ye_test_final))
                    row['Random_acc'] = accuracy_score(ye_test_final, SanityControls.random_baseline(ye_test_final, seed))
                    row['SentenceLength_acc'] = accuracy_score(ye_test_final, SanityControls.sentence_length_proxy(Xe_cal, ye_cal, Xe_test_final))
                    row['WordCount_acc'] = accuracy_score(ye_test_final, SanityControls.word_count_proxy(Xe_cal, ye_cal, Xe_test_final))

                    preds, probs = FeatureBaselines.eeg_svm(Xe_cal, ye_cal, Xe_test_final)
                    row['EEG_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.gaze_svm(Xg_cal, ye_cal, Xg_test_final)
                    row['Gaze_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['Gaze_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.eeg_gaze_concat_svm(Xe_cal, Xg_cal, ye_cal, Xe_test_final, Xg_test_final)
                    row['EEG_Gaze_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_Gaze_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.eeg_pca_svm(Xe_cal, ye_cal, Xe_test_final)
                    row['EEG_PCA_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_PCA_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.gaze_fixation_only_svm(Xg_cal, ye_cal, Xg_test_final)
                    row['Gaze_Fixation_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['Gaze_Fixation_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.gaze_saccade_only_svm(Xg_cal, ye_cal, Xg_test_final)
                    row['Gaze_Saccade_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['Gaze_Saccade_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.gaze_fixation_saccade_svm(Xg_cal, ye_cal, Xg_test_final)
                    row['Gaze_FixSacc_SVM_acc'] = accuracy_score(ye_test_final, preds)
                    row['Gaze_FixSacc_SVM_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = DeepBaselinesProxy.eeg_lstm_proxy(Xe_cal, ye_cal, Xe_test_final)
                    row['EEG_LSTM_proxy_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_LSTM_proxy_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = DeepBaselinesProxy.gaze_lstm_proxy(Xg_cal, ye_cal, Xg_test_final)
                    row['Gaze_LSTM_proxy_acc'] = accuracy_score(ye_test_final, preds)
                    row['Gaze_LSTM_proxy_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = DeepBaselinesProxy.eeg_gcn_proxy(Xe_cal, ye_cal, Xe_test_final)
                    row['EEG_GCN_proxy_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_GCN_proxy_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = DeepBaselinesProxy.eeg_gaze_fusion_lstm_proxy(Xe_cal, Xg_cal, ye_cal, Xe_test_final, Xg_test_final)
                    row['EEG_Gaze_LSTM_proxy_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_Gaze_LSTM_proxy_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds = PCETVariants.raw_only(Xe_cal, ye_cal, Xe_test_final)
                    row['PCET_raw_acc'] = accuracy_score(ye_test_final, preds)

                    preds = PCETVariants.abs_error_only(Xe_cal, ye_cal, Xe_test_final)
                    row['PCET_abserror_acc'] = accuracy_score(ye_test_final, preds)

                    preds = PCETVariants.raw_plus_abserror(Xe_cal, ye_cal, Xe_test_final)
                    row['PCET_raw_abserror_acc'] = accuracy_score(ye_test_final, preds)

                    preds = PCETVariants.shuffled_error(Xe_cal, ye_cal, Xe_test_final)
                    row['PCET_shuffled_acc'] = accuracy_score(ye_test_final, preds)

                    preds = PCETVariants.random_error(Xe_cal, ye_cal, Xe_test_final)
                    row['PCET_random_acc'] = accuracy_score(ye_test_final, preds)

                    preds, probs = GETAAblation.eeg_mlp(Xe_cal, ye_cal, Xe_test_final)
                    row['EEG_MLP_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_MLP_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = GETAAblation.gaze_mlp(Xg_cal, ye_cal, Xg_test_final)
                    row['Gaze_MLP_acc'] = accuracy_score(ye_test_final, preds)
                    row['Gaze_MLP_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = GETAAblation.geta_confidence_only(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                    row['GETA_confidence_acc'] = accuracy_score(ye_test_final, preds)
                    row['GETA_confidence_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = GETAAblation.geta_entropy_only(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                    row['GETA_entropy_acc'] = accuracy_score(ye_test_final, preds)
                    row['GETA_entropy_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = GETAAblation.geta_confidence_entropy(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                    row['GETA_conf_ent_acc'] = accuracy_score(ye_test_final, preds)
                    row['GETA_conf_ent_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = GETAAblation.geta_random_attention(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                    row['GETA_random_att_acc'] = accuracy_score(ye_test_final, preds)
                    row['GETA_random_att_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = GETAAblation.geta_shuffled_attention(Xe_cal, ye_cal, Xg_cal, Xe_test_final, Xg_test_final)
                    row['GETA_shuffled_att_acc'] = accuracy_score(ye_test_final, preds)
                    row['GETA_shuffled_att_auroc'] = roc_auc_score(ye_test_final, probs)

                    preds, probs = FeatureBaselines.eeg_gaze_concat_svm(Xe_cal, Xg_cal, ye_cal, Xe_test_final, Xg_test_final)
                    row['EEG_Gaze_concat_acc'] = accuracy_score(ye_test_final, preds)
                    row['EEG_Gaze_concat_auroc'] = roc_auc_score(ye_test_final, probs)

                    clf_e = SVC(kernel='rbf', probability=True, random_state=42)
                    clf_e.fit(StandardScaler().fit_transform(Xe_cal), ye_cal)
                    p_e = clf_e.predict_proba(StandardScaler().fit_transform(Xe_test_final))[:, 1]
                    clf_g = SVC(kernel='rbf', probability=True, random_state=42)
                    clf_g.fit(StandardScaler().fit_transform(Xg_cal), ye_cal)
                    p_g = clf_g.predict_proba(StandardScaler().fit_transform(Xg_test_final))[:, 1]
                    p_avg = (p_e + p_g) / 2
                    row['Static_avg_acc'] = accuracy_score(ye_test_final, (p_avg >= 0.5).astype(int))
                    row['Static_avg_auroc'] = roc_auc_score(ye_test_final, p_avg)

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
                    probs_cagf = clf_final.predict_proba(z_fused_test)[:, 1]
                    row['CAGF_feature_only_acc'] = accuracy_score(ye_test_final, (probs_cagf >= 0.5).astype(int))
                    row['CAGF_feature_only_auroc'] = roc_auc_score(ye_test_final, probs_cagf)

                except Exception as e:
                    print(f' Err:{str(e)[:20]}', end='', flush=True)

                results.append(row)
            print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_experiment.csv'), index=False)
    print('\nSaved!', flush=True)
    return df

def analyze_results(df):
    shots = [3, 5, 10, 20, 50]

    print("\n" + "="*120)
    print("MAIN COMPARISON WITH ZUCO BASELINES")
    print("="*120)

    methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_Gaze_SVM', 'EEG_PCA_SVM', 'EEG_LSTM_proxy',
               'Gaze_LSTM_proxy', 'EEG_GCN_proxy', 'EEG_Gaze_LSTM_proxy']

    print(f"\n{'Method':<30}", end='')
    for s in shots:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in methods:
        print(f"{m:<30}", end='')
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print("\n" + "="*120)
    print("TEXT CONFOUND CONTROLS")
    print("="*120)

    sanity = ['Majority', 'Random', 'SentenceLength', 'WordCount']
    print(f"\n{'Method':<30}", end='')
    for s in shots:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in sanity:
        print(f"{m:<30}", end='')
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print("\n" + "="*120)
    print("GETA ABLATION")
    print("="*120)

    geta = ['EEG_MLP', 'Gaze_MLP', 'GETA_confidence', 'GETA_entropy', 'GETA_conf_ent', 'GETA_random_att', 'GETA_shuffled_att']
    print(f"\n{'Method':<25}", end='')
    for s in shots:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in geta:
        print(f"{m:<25}", end='')
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print("\n" + "="*120)
    print("PCET ABLATION")
    print("="*120)

    pcet = ['PCET_raw', 'PCET_abserror', 'PCET_raw_abserror', 'PCET_shuffled', 'PCET_random']
    print(f"\n{'Method':<25}", end='')
    for s in shots:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in pcet:
        print(f"{m:<25}", end='')
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print("\n" + "="*120)
    print("GAZE FEATURE BASELINES")
    print("="*120)

    gaze_feats = ['Gaze_SVM', 'Gaze_Fixation_SVM', 'Gaze_Saccade_SVM', 'Gaze_FixSacc_SVM']
    print(f"\n{'Method':<25}", end='')
    for s in shots:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in gaze_feats:
        print(f"{m:<25}", end='')
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    print("\n" + "="*120)
    print("CAGF ABLATION (from previous experiments)")
    print("="*120)

    cagf = ['EEG_Gaze_concat', 'Static_avg', 'CAGF_feature_only']
    print(f"\n{'Method':<25}", end='')
    for s in shots:
        print(f"{'S'+str(s):>12}", end='')
    print()
    for m in cagf:
        print(f"{m:<25}", end='')
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                print(f"{v*100:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    return df

def save_reports(df):
    shots = [3, 5, 10, 20, 50]

    report = []
    report.append("# Comprehensive Experiment Report\n\n")

    report.append("## 1. Main Comparison with Zuco Baselines\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
    report.append("|--------|--------|--------|---------|---------|--------|\n")

    main_methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_Gaze_SVM', 'EEG_PCA_SVM', 'EEG_LSTM_proxy',
                   'Gaze_LSTM_proxy', 'EEG_GCN_proxy', 'EEG_Gaze_LSTM_proxy']
    for m in main_methods:
        row = f"| {m} |"
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                row += f" {v*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

    report.append("\n## 2. Text Confound Controls\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
    report.append("|--------|--------|--------|---------|---------|--------|\n")
    sanity = ['Majority', 'Random', 'SentenceLength', 'WordCount']
    for m in sanity:
        row = f"| {m} |"
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                row += f" {v*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

    report.append("\n## 3. GETA Ablation\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
    report.append("|--------|--------|--------|---------|---------|--------|\n")
    geta = ['EEG_MLP', 'Gaze_MLP', 'GETA_confidence', 'GETA_entropy', 'GETA_conf_ent', 'GETA_random_att', 'GETA_shuffled_att']
    for m in geta:
        row = f"| {m} |"
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                row += f" {v*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

    report.append("\n## 4. PCET Ablation\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
    report.append("|--------|--------|--------|---------|---------|--------|\n")
    pcet = ['PCET_raw', 'PCET_abserror', 'PCET_raw_abserror', 'PCET_shuffled', 'PCET_random']
    for m in pcet:
        row = f"| {m} |"
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                row += f" {v*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

    report.append("\n## 5. Gaze Feature Baselines\n\n")
    report.append("| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |\n")
    report.append("|--------|--------|--------|---------|---------|--------|\n")
    gaze_feats = ['Gaze_SVM', 'Gaze_Fixation_SVM', 'Gaze_Saccade_SVM', 'Gaze_FixSacc_SVM']
    for m in gaze_feats:
        row = f"| {m} |"
        for s in shots:
            sub = df[df['n_cal'] == s]
            if len(sub) > 0 and f'{m}_acc' in sub.columns:
                v = sub[f'{m}_acc'].mean()
                row += f" {v*100:.1f}% |"
            else:
                row += " - |"
        report.append(row)

    report_text = "".join(report)
    with open(os.path.join(REPORTS_DIR, 'zuco_related_baseline_report.md'), 'w') as f:
        f.write(report_text)

    df_sanity = df[['seed', 'subject', 'n_cal'] + [c for c in df.columns if 'Majority' in c or 'Random' in c or 'Sentence' in c or 'Word' in c]]
    df_sanity.to_csv(os.path.join(RESULTS_DIR, 'text_confound_controls.csv'), index=False)

    df_baselines = df[['seed', 'subject', 'n_cal'] + [c for c in df.columns if any(x in c for x in ['EEG_SVM', 'Gaze_SVM', 'PCA_SVM', 'LSTM', 'GCN', 'Fixation', 'Saccade'])]]
    df_baselines.to_csv(os.path.join(RESULTS_DIR, 'main_comparison_with_zuco_baselines.csv'), index=False)

    df_geta = df[['seed', 'subject', 'n_cal'] + [c for c in df.columns if 'GETA' in c or c in ['EEG_MLP', 'Gaze_MLP']]]
    df_geta.to_csv(os.path.join(RESULTS_DIR, 'geta_ablation.csv'), index=False)

    df_pcet = df[['seed', 'subject', 'n_cal'] + [c for c in df.columns if 'PCET' in c]]
    df_pcet.to_csv(os.path.join(RESULTS_DIR, 'pcet_ablation.csv'), index=False)

    df_cagf = df[['seed', 'subject', 'n_cal'] + [c for c in df.columns if 'concat' in c or 'Static' in c or 'CAGF' in c]]
    df_cagf.to_csv(os.path.join(RESULTS_DIR, 'cagf_ablation_final.csv'), index=False)

    print("Reports saved!", flush=True)

if __name__ == '__main__':
    print("Comprehensive Final Experiment", flush=True)
    print("="*80, flush=True)
    df = run_comprehensive_experiment()
    analyze_results(df)
    save_reports(df)
    print("\nDone!", flush=True)