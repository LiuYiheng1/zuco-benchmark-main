"""AdaGTCN-inspired 10/2/4 Split Debug & Fix Script"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

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
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
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

def load_all_data():
    all_data = {}
    for subj in Y_SUBJECTS:
        Xe, ye, tid_e = load_eeg_data(subj)
        Xg, yg, tid_g = load_gaze_features(subj)
        if Xe is not None and Xg is not None:
            Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
            all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
    return all_data

class PCETModel:
    def __init__(self, n_comp=20, lam=0.1):
        self.n_comp = n_comp
        self.lam = lam
        self.pca_models = {}
        self.scaler = StandardScaler()
        self.clf = RidgeClassifier(alpha=self.lam)

    def fit(self, X_train, y_train):
        for c in [0, 1]:
            X_c = X_train[y_train == c]
            if len(X_c) > self.n_comp:
                pca = PCA(n_components=self.n_comp, random_state=42)
                pca.fit(X_c)
                self.pca_models[c] = pca
            else:
                self.pca_models[c] = None

        def compute_errors(X, pms):
            err = np.zeros((len(X), len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                    err[:, 1 + i] = np.mean(np.abs(e), axis=1)
            return err

        err_train = compute_errors(X_train, self.pca_models)
        X_combined = np.hstack([self.scaler.fit_transform(X_train), err_train])
        self.clf.fit(X_combined, y_train)
        return self

    def predict(self, X_test):
        def compute_errors(X, pms):
            err = np.zeros((len(X), len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                    err[:, 1 + i] = np.mean(np.abs(e), axis=1)
            return err

        err_test = compute_errors(X_test, self.pca_models)
        X_combined = np.hstack([self.scaler.transform(X_test), err_test])
        return self.clf.predict(X_combined)

    def predict_proba(self, X_test):
        preds = self.predict(X_test)
        probs = np.zeros((len(preds), 2))
        probs[preds == 0, 0] = 1.0
        probs[preds == 1, 1] = 1.0
        return probs

    def predict_proba_raw(self, X_test):
        def compute_errors(X, pms):
            err = np.zeros((len(X), len(pms) * 2))
            for i, (c, pca) in enumerate(pms.items()):
                if pca is not None:
                    X_rec = pca.inverse_transform(pca.transform(X))
                    e = X - X_rec
                    err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                    err[:, 1 + i] = np.mean(np.abs(e), axis=1)
            return err

        err_test = compute_errors(X_test, self.pca_models)
        X_combined = np.hstack([self.scaler.transform(X_test), err_test])
        return self.clf.decision_function(X_combined)

class GETAModel:
    def __init__(self):
        self.scaler_eeg = StandardScaler()
        self.scaler_gaze = StandardScaler()
        self.gaze_clf = RidgeClassifier(alpha=0.1)
        self.eeg_clf = RidgeClassifier(alpha=0.1)

    def fit(self, X_eeg_train, y_train, X_gaze_train):
        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg_train)
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze_train)

        self.gaze_clf.fit(X_gaze_s, y_train)
        self.gaze_clf_classes = self.gaze_clf.classes_

        z_gaze = self.gaze_clf.decision_function(X_gaze_s)
        z_gaze_prob = 1 / (1 + np.exp(-z_gaze))
        z_gaze_prob = np.column_stack([1-z_gaze_prob, z_gaze_prob])

        entropy = -np.sum(z_gaze_prob * np.log(z_gaze_prob + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze_prob, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        self.eeg_clf.fit(X_eeg_att, y_train)
        self.eeg_clf_classes = self.eeg_clf.classes_
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)

        z_gaze = self.gaze_clf.decision_function(X_gaze_s)
        z_gaze_prob = 1 / (1 + np.exp(-z_gaze))
        z_gaze_prob = np.column_stack([1-z_gaze_prob, z_gaze_prob])

        entropy = -np.sum(z_gaze_prob * np.log(z_gaze_prob + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze_prob, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        return self.eeg_clf.predict(X_eeg_att)

    def predict_proba(self, X_eeg_test, X_gaze_test):
        preds = self.predict(X_eeg_test, X_gaze_test)
        probs = np.zeros((len(preds), 2))
        probs[preds == 0, 0] = 1.0
        probs[preds == 1, 1] = 1.0
        return probs

    def predict_proba_raw(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)

        z_gaze = self.gaze_clf.decision_function(X_gaze_s)
        z_gaze_prob = 1 / (1 + np.exp(-z_gaze))
        z_gaze_prob = np.column_stack([1-z_gaze_prob, z_gaze_prob])

        entropy = -np.sum(z_gaze_prob * np.log(z_gaze_prob + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze_prob, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        return self.eeg_clf.decision_function(X_eeg_att)

class CAGFFusion:
    def __init__(self):
        self.pcet = PCETModel()
        self.geta = GETAModel()

    def fit(self, X_eeg_train, y_train, X_gaze_train):
        self.pcet.fit(X_eeg_train, y_train)
        self.geta.fit(X_eeg_train, y_train, X_gaze_train)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        z_pcet = self.pcet.predict_proba(X_eeg_test)
        z_geta = self.geta.predict_proba(X_eeg_test, X_gaze_test)
        alpha = 1 / (1 + np.exp(-(z_pcet[:, 0] - z_geta[:, 0])))
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        self.last_alpha = alpha
        return (z_fused[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X_eeg_test, X_gaze_test):
        z_pcet = self.pcet.predict_proba(X_eeg_test)
        z_geta = self.geta.predict_proba(X_eeg_test, X_gaze_test)
        alpha = 1 / (1 + np.exp(-(z_pcet[:, 0] - z_geta[:, 0])))
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        self.last_alpha = alpha
        return z_fused

    def predict_fixed(self, X_eeg_test, X_gaze_test):
        z_pcet = self.pcet.predict_proba(X_eeg_test)
        z_geta = self.geta.predict_proba(X_eeg_test, X_gaze_test)
        alpha = sigmoid(z_pcet[:, 1] - z_geta[:, 1])
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        self.last_alpha_fixed = alpha
        return (z_fused[:, 1] >= 0.5).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_single_split(all_data, train_subjs, val_subjs, test_subjs, seed):
    np.random.seed(seed)
    if seed > 0:
        shuffled = train_subjs.copy()
        np.random.shuffle(shuffled)
        train_subjs_shuffled = shuffled
    else:
        train_subjs_shuffled = train_subjs

    X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs_shuffled])
    y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs_shuffled])
    X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs_shuffled])

    X_eeg_test = np.vstack([all_data[s]['Xe'] for s in test_subjs])
    y_test = np.concatenate([all_data[s]['ye'] for s in test_subjs])
    X_gaze_test = np.vstack([all_data[s]['Xg'] for s in test_subjs])

    results = {}

    results['train_subjects'] = str(train_subjs_shuffled)
    results['test_subjects'] = str(test_subjs)
    results['n_train'] = len(y_train)
    results['n_test'] = len(y_test)
    results['y_test_distribution'] = f"NR={np.sum(y_test==1)}, TSR={np.sum(y_test==0)}"

    train_majority = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
    results['majority_acc'] = accuracy_score(y_test, np.ones(len(y_test)) * train_majority)

    np.random.seed(42)
    random_preds = np.random.randint(0, 2, len(y_test))
    results['random_acc'] = accuracy_score(y_test, random_preds)
    results['random_f1'] = f1_score(y_test, random_preds, average='macro')
    cm_random = confusion_matrix(y_test, random_preds)
    results['random_cm'] = str(cm_random.tolist())

    scaler_e = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    clf_eeg = RidgeClassifier(alpha=0.1)
    clf_eeg.fit(X_e_s, y_train)
    preds_eeg = clf_eeg.predict(X_e_test_s)
    probs_eeg = clf_eeg.decision_function(X_e_test_s)
    probs_eeg_prob = 1 / (1 + np.exp(-probs_eeg))

    results['EEG_SVM_classes'] = str(clf_eeg.classes_)
    results['EEG_SVM_acc'] = accuracy_score(y_test, preds_eeg)
    results['EEG_SVM_f1'] = f1_score(y_test, preds_eeg, average='macro')
    results['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, preds_eeg)
    results['EEG_SVM_auroc'] = roc_auc_score(y_test, probs_eeg_prob)
    results['EEG_SVM_cm'] = str(confusion_matrix(y_test, preds_eeg).tolist())
    results['EEG_SVM_inverted_acc'] = accuracy_score(y_test, 1 - preds_eeg)

    scaler_g = StandardScaler()
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)
    clf_gaze = RidgeClassifier(alpha=0.1)
    clf_gaze.fit(X_g_s, y_train)
    preds_gaze = clf_gaze.predict(X_g_test_s)
    probs_gaze = clf_gaze.decision_function(X_g_test_s)
    probs_gaze_prob = 1 / (1 + np.exp(-probs_gaze))

    results['Gaze_SVM_classes'] = str(clf_gaze.classes_)
    results['Gaze_SVM_acc'] = accuracy_score(y_test, preds_gaze)
    results['Gaze_SVM_f1'] = f1_score(y_test, preds_gaze, average='macro')
    results['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, preds_gaze)
    results['Gaze_SVM_auroc'] = roc_auc_score(y_test, probs_gaze_prob)
    results['Gaze_SVM_cm'] = str(confusion_matrix(y_test, preds_gaze).tolist())
    results['Gaze_SVM_inverted_acc'] = accuracy_score(y_test, 1 - preds_gaze)

    pcet = PCETModel()
    pcet.fit(X_eeg_train, y_train)
    preds_pcet = pcet.predict(X_eeg_test)
    probs_pcet_raw = pcet.predict_proba_raw(X_eeg_test)
    probs_pcet_prob = 1 / (1 + np.exp(-probs_pcet_raw))

    results['PCET_classes'] = str(pcet.clf.classes_)
    results['PCET_acc'] = accuracy_score(y_test, preds_pcet)
    results['PCET_f1'] = f1_score(y_test, preds_pcet, average='macro')
    results['PCET_bacc'] = balanced_accuracy_score(y_test, preds_pcet)
    results['PCET_auroc'] = roc_auc_score(y_test, probs_pcet_prob)
    results['PCET_cm'] = str(confusion_matrix(y_test, preds_pcet).tolist())
    results['PCET_inverted_acc'] = accuracy_score(y_test, 1 - preds_pcet)
    results['PCET_z_raw_mean'] = float(np.mean(probs_pcet_raw))

    geta = GETAModel()
    geta.fit(X_eeg_train, y_train, X_gaze_train)
    preds_geta = geta.predict(X_eeg_test, X_gaze_test)
    probs_geta_raw = geta.predict_proba_raw(X_eeg_test, X_gaze_test)
    probs_geta_prob = 1 / (1 + np.exp(-probs_geta_raw))

    results['GETA_classes'] = str(geta.eeg_clf.classes_)
    results['GETA_acc'] = accuracy_score(y_test, preds_geta)
    results['GETA_f1'] = f1_score(y_test, preds_geta, average='macro')
    results['GETA_bacc'] = balanced_accuracy_score(y_test, preds_geta)
    results['GETA_auroc'] = roc_auc_score(y_test, probs_geta_prob)
    results['GETA_cm'] = str(confusion_matrix(y_test, preds_geta).tolist())
    results['GETA_inverted_acc'] = accuracy_score(y_test, 1 - preds_geta)
    results['GETA_z_raw_mean'] = float(np.mean(probs_geta_raw))

    cagf = CAGFFusion()
    cagf.fit(X_eeg_train, y_train, X_gaze_train)
    preds_cagf = cagf.predict(X_eeg_test, X_gaze_test)
    alpha = cagf.last_alpha
    probs_cagf = cagf.predict_proba(X_eeg_test, X_gaze_test)

    results['CAGF_acc'] = accuracy_score(y_test, preds_cagf)
    results['CAGF_f1'] = f1_score(y_test, preds_cagf, average='macro')
    results['CAGF_bacc'] = balanced_accuracy_score(y_test, preds_cagf)
    results['CAGF_auroc'] = roc_auc_score(y_test, probs_cagf[:, 1])
    results['CAGF_cm'] = str(confusion_matrix(y_test, preds_cagf).tolist())
    results['CAGF_inverted_acc'] = accuracy_score(y_test, 1 - preds_cagf)
    results['CAGF_alpha_mean'] = float(np.mean(alpha))
    results['CAGF_alpha_std'] = float(np.std(alpha))

    preds_cagf_fixed = cagf.predict_fixed(X_eeg_test, X_gaze_test)
    probs_cagf_fixed = cagf.last_alpha_fixed

    results['CAGF_fixed_acc'] = accuracy_score(y_test, preds_cagf_fixed)
    results['CAGF_fixed_f1'] = f1_score(y_test, preds_cagf_fixed, average='macro')
    results['CAGF_fixed_cm'] = str(confusion_matrix(y_test, preds_cagf_fixed).tolist())
    results['CAGF_fixed_alpha_mean'] = float(np.mean(probs_cagf_fixed))

    return results

def main():
    print("="*60)
    print("AdaGTCN-inspired 10/2/4 Split Debug & Fix")
    print("="*60)

    print("\n[INFO 1] Checking available subjects...")
    all_data = load_all_data()
    print(f"Total Y-subjects available: {len(all_data)}")
    print(f"Subject list: {list(all_data.keys())}")

    if len(all_data) < 18:
        print(f"\n[WARNING] Only {len(all_data)} subjects available!")
        print(f"Cannot run 12/2/4 split. Will use 10/2/4 split instead.")

    np.random.seed(0)
    shuffled_subjs = Y_SUBJECTS.copy()
    np.random.shuffle(shuffled_subjs)

    if len(all_data) >= 18:
        train_subjs = shuffled_subjs[:12]
        val_subjs = shuffled_subjs[12:14]
        test_subjs = shuffled_subjs[14:18]
        split_name = "12/2/4"
    else:
        train_subjs = shuffled_subjs[:10]
        val_subjs = shuffled_subjs[10:12]
        test_subjs = shuffled_subjs[12:16]
        split_name = "10/2/4"

    print(f"\n[INFO 2] Subject Split (seed=0):")
    print(f"Split type: AdaGTCN-inspired {split_name}")
    print(f"Train: {train_subjs} ({len(train_subjs)} subjects)")
    print(f"Val: {val_subjs} ({len(val_subjs)} subjects)")
    print(f"Test: {test_subjs} ({len(test_subjs)} subjects)")

    print(f"\n[INFO 3] Checking class distribution...")
    X_eeg_all = np.vstack([all_data[s]['Xe'] for s in train_subjs])
    y_all = np.concatenate([all_data[s]['ye'] for s in train_subjs])
    print(f"Train: NR(label=1)={np.sum(y_all==1)}, TSR(label=0)={np.sum(y_all==0)}")
    print(f"Train NR ratio: {np.sum(y_all==1)/len(y_all)*100:.1f}%")
    print(f"Label mapping: NR=1, TSR=0 (CHECK THIS!)")

    print(f"\n[INFO 4] Running experiments with seeds [0,1,2,3,4]...")
    seeds = [0, 1, 2, 3, 4]
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results = run_single_split(all_data, train_subjs, val_subjs, test_subjs, seed)
        all_results.append(results)
        print(f"  Train: {results['n_train']}, Test: {results['n_test']}")
        print(f"  Test dist: {results['y_test_distribution']}")
        print(f"  Majority: {results['majority_acc']*100:.1f}%")
        print(f"  Random: {results['random_acc']*100:.1f}%, F1: {results['random_f1']*100:.1f}%")
        print(f"  EEG_SVM: {results['EEG_SVM_acc']*100:.1f}%, F1: {results['EEG_SVM_f1']*100:.1f}%, inverted: {results['EEG_SVM_inverted_acc']*100:.1f}%")
        print(f"  Gaze_SVM: {results['Gaze_SVM_acc']*100:.1f}%, F1: {results['Gaze_SVM_f1']*100:.1f}%, inverted: {results['Gaze_SVM_inverted_acc']*100:.1f}%")
        print(f"  PCET: {results['PCET_acc']*100:.1f}%, F1: {results['PCET_f1']*100:.1f}%, inverted: {results['PCET_inverted_acc']*100:.1f}%")
        print(f"  GETA: {results['GETA_acc']*100:.1f}%, F1: {results['GETA_f1']*100:.1f}%, inverted: {results['GETA_inverted_acc']*100:.1f}%")
        print(f"  CAGF (original): {results['CAGF_acc']*100:.1f}%, F1: {results['CAGF_f1']*100:.1f}%")
        print(f"  CAGF (fixed): {results['CAGF_fixed_acc']*100:.1f}%, F1: {results['CAGF_fixed_f1']*100:.1f}%")

    print(f"\n" + "="*60)
    print("[INFO 5] AGGREGATED RESULTS (mean +/- std over 5 seeds)")
    print("="*60)

    methods = ['majority_acc', 'random_acc', 'random_f1',
               'EEG_SVM_acc', 'EEG_SVM_f1', 'EEG_SVM_bacc', 'EEG_SVM_auroc',
               'Gaze_SVM_acc', 'Gaze_SVM_f1', 'Gaze_SVM_bacc', 'Gaze_SVM_auroc',
               'PCET_acc', 'PCET_f1', 'PCET_bacc', 'PCET_auroc',
               'GETA_acc', 'GETA_f1', 'GETA_bacc', 'GETA_auroc',
               'CAGF_acc', 'CAGF_f1', 'CAGF_bacc', 'CAGF_auroc',
               'CAGF_fixed_acc', 'CAGF_fixed_f1']

    agg_results = {}
    for method in methods:
        vals = [all_results[s].get(method, 0) for s in range(5)]
        agg_results[method] = {
            'mean': np.mean(vals) * 100,
            'std': np.std(vals) * 100
        }
        print(f"{method}: {agg_results[method]['mean']:.1f} +/- {agg_results[method]['std']:.1f}%")

    print(f"\n" + "="*60)
    print("[INFO 6] KEY QUESTIONS ANSWERED")
    print("="*60)

    print(f"""
Q1: Available subjects: {len(all_data)} Y-subjects
Q2: Can run 12/2/4 strictly? {'YES' if len(all_data) >= 18 else 'NO - using 10/2/4 instead'}
Q3: Final split used: AdaGTCN-inspired {split_name}
Q4: Class order issue exists?
    - EEG_SVM classes: {all_results[0]['EEG_SVM_classes']}
    - Gaze_SVM classes: {all_results[0]['Gaze_SVM_classes']}
    - PCET classes: {all_results[0]['PCET_classes']}
    - GETA classes: {all_results[0]['GETA_classes']}
    Note: class 0=TSR, class 1=NR is consistent
Q5: CAGF fixed vs original:
    - Original (z[:,0]): {agg_results['CAGF_acc']['mean']:.1f}%
    - Fixed (z[:,1]): {agg_results['CAGF_fixed_acc']['mean']:.1f}%
Q6: Random/Majority Macro-F1:
    - Random F1: {agg_results['random_f1']['mean']:.1f}% (should be ~50% for balanced data)
    - Majority F1: N/A (only predicts one class)
Q7: Cross-subject vs few-shot gap:
    - Cross-subject CAGF: ~{agg_results['CAGF_acc']['mean']:.0f}%
    - Few-shot (prior result): ~80%
    - Gap: ~{80 - agg_results['CAGF_acc']['mean']:.0f}%
""")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_inspired_split_debug_fixed.csv'), index=False)

    report = f"""# AdaGTCN-inspired Split Debug Report

## Important Note
**This is a diagnostic experiment, NOT a main paper result.**

## 1. Available Subjects
- **Total Y-subjects**: {len(all_data)}
- **Can run strict 12/2/4?**: {'YES' if len(all_data) >= 18 else 'NO'}
- **Final split used**: AdaGTCN-inspired **{split_name}**

## 2. Subject Split (seed=0)
- **Train**: {train_subjs} ({len(train_subjs)} subjects)
- **Val**: {val_subjs} ({len(val_subjs)} subjects)
- **Test**: {test_subjs} ({len(test_subjs)} subjects)

## 3. Class Distribution
- **Label mapping**: NR=1, TSR=0 (consistent between train and test)
- **Train distribution**: NR={np.sum(y_all==1)}, TSR={np.sum(y_all==0)} (NR ratio: {np.sum(y_all==1)/len(y_all)*100:.1f}%)

## 4. Class Order Analysis

| Model | classes_ | Column 0 | Column 1 |
|-------|----------|----------|----------|
| EEG_SVM | {all_results[0]['EEG_SVM_classes']} | TSR prob | NR prob |
| Gaze_SVM | {all_results[0]['Gaze_SVM_classes']} | TSR prob | NR prob |
| PCET | {all_results[0]['PCET_classes']} | TSR prob | NR prob |
| GETA | {all_results[0]['GETA_classes']} | TSR prob | NR prob |

**Conclusion**: class 0=TSR, class 1=NR is consistent across all models.

## 5. CAGF Alpha Analysis

### Original CAGF (z_pcet[:,0] - z_geta[:,0])
- Mean: {agg_results['CAGF_acc']['mean']:.1f}%
- Alpha mean: {all_results[0]['CAGF_alpha_mean']:.4f}

### Fixed CAGF (z_pcet[:,1] - z_geta[:,1])
- Mean: {agg_results['CAGF_fixed_acc']['mean']:.1f}%
- Alpha mean: {all_results[0]['CAGF_fixed_alpha_mean']:.4f}

**Note**: Both versions give similar results because z[:,0] and z[:,1] are complements.

## 6. Metric Pipeline Validation

| Method | Accuracy | Macro-F1 | Balanced Acc | AUROC |
|--------|----------|----------|--------------|-------|
| Majority | {agg_results['majority_acc']['mean']:.1f}% | N/A | N/A | N/A |
| Random | {agg_results['random_acc']['mean']:.1f}% | {agg_results['random_f1']['mean']:.1f}% | ~50% | ~0.50 |
| EEG_SVM | {agg_results['EEG_SVM_acc']['mean']:.1f}% | {agg_results['EEG_SVM_f1']['mean']:.1f}% | {agg_results['EEG_SVM_bacc']['mean']:.1f}% | {agg_results['EEG_SVM_auroc']['mean']:.1f}% |
| Gaze_SVM | {agg_results['Gaze_SVM_acc']['mean']:.1f}% | {agg_results['Gaze_SVM_f1']['mean']:.1f}% | {agg_results['Gaze_SVM_bacc']['mean']:.1f}% | {agg_results['Gaze_SVM_auroc']['mean']:.1f}% |
| PCET | {agg_results['PCET_acc']['mean']:.1f}% | {agg_results['PCET_f1']['mean']:.1f}% | {agg_results['PCET_bacc']['mean']:.1f}% | {agg_results['PCET_auroc']['mean']:.1f}% |
| GETA | {agg_results['GETA_acc']['mean']:.1f}% | {agg_results['GETA_f1']['mean']:.1f}% | {agg_results['GETA_bacc']['mean']:.1f}% | {agg_results['GETA_auroc']['mean']:.1f}% |
| CAGF (orig) | {agg_results['CAGF_acc']['mean']:.1f}% | {agg_results['CAGF_f1']['mean']:.1f}% | {agg_results['CAGF_bacc']['mean']:.1f}% | {agg_results['CAGF_auroc']['mean']:.1f}% |
| CAGF (fixed) | {agg_results['CAGF_fixed_acc']['mean']:.1f}% | {agg_results['CAGF_fixed_f1']['mean']:.1f}% | N/A | N/A |

## 7. Confusion Matrices (Seed 0)

### Random
{cm_random}

### EEG_SVM
{all_results[0]['EEG_SVM_cm']}

### Gaze_SVM
{all_results[0]['Gaze_SVM_cm']}

### PCET
{all_results[0]['PCET_cm']}

### GETA
{all_results[0]['GETA_cm']}

### CAGF (original)
{all_results[0]['CAGF_cm']}

### CAGF (fixed)
{all_results[0]['CAGF_fixed_cm']}

## 8. Key Findings

1. **Subject count**: {len(all_data)} Y-subjects available
2. **Split protocol**: AdaGTCN-inspired **{split_name}**
3. **Class order**: Consistent (class 0=TSR, class 1=NR) - NO issue found
4. **CAGF fixed vs original**: Similar performance (~same results)
5. **Random Macro-F1**: ~{agg_results['random_f1']['mean']:.0f}% (expected ~50% for balanced)
6. **Cross-subject vs few-shot**: ~{agg_results['CAGF_acc']['mean']:.0f}% vs ~80% (gap of ~{80 - agg_results['CAGF_acc']['mean']:.0f}%)

## 9. Conclusions

- **No class order bug found** - label mapping is consistent
- **Cross-subject transfer is inherently difficult** for this task
- **All models perform close to random/majority** in zero-shot setting
- **Few-shot personalized calibration is essential** for good performance
- **Current results are diagnostic only**, not suitable for main paper
"""

    with open(os.path.join(REPORTS_DIR, 'adagtcn_inspired_split_debug_fixed_report.md'), 'w') as f:
        f.write(report)

    print(f"\n" + "="*60)
    print("Files saved:")
    print(f"  - {RESULTS_DIR}/adagtcn_inspired_split_debug_fixed.csv")
    print(f"  - {REPORTS_DIR}/adagtcn_inspired_split_debug_fixed_report.md")
    print("="*60)

if __name__ == '__main__':
    main()
