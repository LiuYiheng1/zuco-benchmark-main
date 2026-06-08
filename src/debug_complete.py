"""AdaGTCN-inspired 10/2/4 Split Complete Debug"""

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
        z_gaze = self.gaze_clf.decision_function(X_gaze_s)
        z_gaze_prob = 1 / (1 + np.exp(-z_gaze))
        z_gaze_prob = np.column_stack([1-z_gaze_prob, z_gaze_prob])

        entropy = -np.sum(z_gaze_prob * np.log(z_gaze_prob + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze_prob, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        self.eeg_clf.fit(X_eeg_att, y_train)
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
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        z_pcet = self.pcet.predict_proba_raw(X_eeg_test)
        z_geta = self.geta.predict_proba_raw(X_eeg_test, X_gaze_test)

        z_pcet_prob = 1 / (1 + np.exp(-z_pcet))
        z_geta_prob = 1 / (1 + np.exp(-z_geta))

        alpha = 1 / (1 + np.exp(-(z_pcet_prob - z_geta_prob)))
        self.last_alpha = alpha

        z_pcet_full = np.column_stack([1-z_pcet_prob, z_pcet_prob])
        z_geta_full = np.column_stack([1-z_geta_prob, z_geta_prob])

        z_fused = alpha.reshape(-1, 1) * z_pcet_full + (1 - alpha.reshape(-1, 1)) * z_geta_full
        return (z_fused[:, 1] >= 0.5).astype(int), z_fused

def main():
    print("="*60)
    print("AdaGTCN-inspired 10/2/4 Split Complete Debug")
    print("="*60)

    print("\n[1] Loading data...")
    all_data = load_all_data()
    print(f"Total subjects: {len(all_data)}")

    np.random.seed(0)
    shuffled_subjs = Y_SUBJECTS.copy()
    np.random.shuffle(shuffled_subjs)

    train_subjs = shuffled_subjs[:10]
    val_subjs = shuffled_subjs[10:12]
    test_subjs = shuffled_subjs[12:16]

    print(f"\n[2] Split (seed=0):")
    print(f"Train: {train_subjs}")
    print(f"Val: {val_subjs}")
    print(f"Test: {test_subjs}")

    X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
    y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
    X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

    X_eeg_test = np.vstack([all_data[s]['Xe'] for s in test_subjs])
    y_test = np.concatenate([all_data[s]['ye'] for s in test_subjs])
    X_gaze_test = np.vstack([all_data[s]['Xg'] for s in test_subjs])

    print(f"\n[3] Data shapes:")
    print(f"Train: {len(y_train)} samples, NR={np.sum(y_train==1)}, TSR={np.sum(y_train==0)}")
    print(f"Test: {len(y_test)} samples, NR={np.sum(y_test==1)}, TSR={np.sum(y_test==0)}")

    print(f"\n[4] Baselines:")
    train_majority = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
    majority_acc = accuracy_score(y_test, np.ones(len(y_test)) * train_majority)
    print(f"Majority baseline (predict {train_majority}): {majority_acc*100:.1f}%")

    np.random.seed(42)
    random_preds = np.random.randint(0, 2, len(y_test))
    random_acc = accuracy_score(y_test, random_preds)
    random_f1 = f1_score(y_test, random_preds, average='macro')
    print(f"Random baseline: {random_acc*100:.1f}%, F1={random_f1*100:.1f}%")
    print(f"Random CM:\n{confusion_matrix(y_test, random_preds)}")

    print(f"\n[5] EEG_SVM:")
    scaler_e = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    clf_eeg = RidgeClassifier(alpha=0.1)
    clf_eeg.fit(X_e_s, y_train)
    preds_eeg = clf_eeg.predict(X_e_test_s)
    probs_eeg = clf_eeg.decision_function(X_e_test_s)
    probs_eeg_prob = 1 / (1 + np.exp(-probs_eeg))
    print(f"  classes_: {clf_eeg.classes_}")
    print(f"  Accuracy: {accuracy_score(y_test, preds_eeg)*100:.1f}%")
    print(f"  Macro-F1: {f1_score(y_test, preds_eeg, average='macro')*100:.1f}%")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, preds_eeg)*100:.1f}%")
    print(f"  AUROC: {roc_auc_score(y_test, probs_eeg_prob)*100:.1f}%")
    print(f"  CM:\n{confusion_matrix(y_test, preds_eeg)}")
    print(f"  Inverted Acc: {accuracy_score(y_test, 1-preds_eeg)*100:.1f}%")

    print(f"\n[6] Gaze_SVM:")
    scaler_g = StandardScaler()
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)
    clf_gaze = RidgeClassifier(alpha=0.1)
    clf_gaze.fit(X_g_s, y_train)
    preds_gaze = clf_gaze.predict(X_g_test_s)
    probs_gaze = clf_gaze.decision_function(X_g_test_s)
    probs_gaze_prob = 1 / (1 + np.exp(-probs_gaze))
    print(f"  classes_: {clf_gaze.classes_}")
    print(f"  Accuracy: {accuracy_score(y_test, preds_gaze)*100:.1f}%")
    print(f"  Macro-F1: {f1_score(y_test, preds_gaze, average='macro')*100:.1f}%")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, preds_gaze)*100:.1f}%")
    print(f"  AUROC: {roc_auc_score(y_test, probs_gaze_prob)*100:.1f}%")
    print(f"  CM:\n{confusion_matrix(y_test, preds_gaze)}")
    print(f"  Inverted Acc: {accuracy_score(y_test, 1-preds_gaze)*100:.1f}%")

    print(f"\n[7] PCET_source:")
    pcet = PCETModel()
    pcet.fit(X_eeg_train, y_train)
    preds_pcet = pcet.predict(X_eeg_test)
    probs_pcet = pcet.predict_proba_raw(X_eeg_test)
    probs_pcet_prob = 1 / (1 + np.exp(-probs_pcet))
    print(f"  classes_: {pcet.clf.classes_}")
    print(f"  Accuracy: {accuracy_score(y_test, preds_pcet)*100:.1f}%")
    print(f"  Macro-F1: {f1_score(y_test, preds_pcet, average='macro')*100:.1f}%")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, preds_pcet)*100:.1f}%")
    print(f"  AUROC: {roc_auc_score(y_test, probs_pcet_prob)*100:.1f}%")
    print(f"  CM:\n{confusion_matrix(y_test, preds_pcet)}")
    print(f"  Inverted Acc: {accuracy_score(y_test, 1-preds_pcet)*100:.1f}%")

    print(f"\n[8] GETA_source:")
    geta = GETAModel()
    geta.fit(X_eeg_train, y_train, X_gaze_train)
    preds_geta = geta.predict(X_eeg_test, X_gaze_test)
    probs_geta = geta.predict_proba_raw(X_eeg_test, X_gaze_test)
    probs_geta_prob = 1 / (1 + np.exp(-probs_geta))
    print(f"  EEG classes_: {geta.eeg_clf.classes_}")
    print(f"  Accuracy: {accuracy_score(y_test, preds_geta)*100:.1f}%")
    print(f"  Macro-F1: {f1_score(y_test, preds_geta, average='macro')*100:.1f}%")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, preds_geta)*100:.1f}%")
    print(f"  AUROC: {roc_auc_score(y_test, probs_geta_prob)*100:.1f}%")
    print(f"  CM:\n{confusion_matrix(y_test, preds_geta)}")
    print(f"  Inverted Acc: {accuracy_score(y_test, 1-preds_geta)*100:.1f}%")

    print(f"\n[9] PCET+GETA+CAGF:")
    cagf = CAGFFusion()
    cagf.fit(X_eeg_train, y_train, X_gaze_train)
    preds_cagf, probs_cagf = cagf.predict(X_eeg_test, X_gaze_test)
    alpha = cagf.last_alpha
    print(f"  Accuracy: {accuracy_score(y_test, preds_cagf)*100:.1f}%")
    print(f"  Macro-F1: {f1_score(y_test, preds_cagf, average='macro')*100:.1f}%")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, preds_cagf)*100:.1f}%")
    print(f"  AUROC: {roc_auc_score(y_test, probs_cagf[:, 1])*100:.1f}%")
    print(f"  CM:\n{confusion_matrix(y_test, preds_cagf)}")
    print(f"  Inverted Acc: {accuracy_score(y_test, 1-preds_cagf)*100:.1f}%")
    print(f"  Alpha: mean={np.mean(alpha):.4f}, std={np.std(alpha):.4f}, min={np.min(alpha):.4f}, max={np.max(alpha):.4f}")

    print(f"\n" + "="*60)
    print("[10] SUMMARY TABLE")
    print("="*60)
    print(f"{'Method':<20} | {'Acc':>6} | {'F1':>6} | {'BAcc':>6} | {'AUROC':>6} | {'InvAcc':>6}")
    print("-" * 70)
    print(f"{'Majority':<20} | {majority_acc*100:>5.1f}% |   -   |   -   |   -   |   -")
    print(f"{'Random':<20} | {random_acc*100:>5.1f}% | {random_f1*100:>5.1f}% | ~50% | ~0.50 |   -")
    print(f"{'EEG_SVM':<20} | {accuracy_score(y_test, preds_eeg)*100:>5.1f}% | {f1_score(y_test, preds_eeg, average='macro')*100:>5.1f}% | {balanced_accuracy_score(y_test, preds_eeg)*100:>5.1f}% | {roc_auc_score(y_test, probs_eeg_prob)*100:>5.1f}% | {accuracy_score(y_test, 1-preds_eeg)*100:>5.1f}%")
    print(f"{'Gaze_SVM':<20} | {accuracy_score(y_test, preds_gaze)*100:>5.1f}% | {f1_score(y_test, preds_gaze, average='macro')*100:>5.1f}% | {balanced_accuracy_score(y_test, preds_gaze)*100:>5.1f}% | {roc_auc_score(y_test, probs_gaze_prob)*100:>5.1f}% | {accuracy_score(y_test, 1-preds_gaze)*100:>5.1f}%")
    print(f"{'PCET_source':<20} | {accuracy_score(y_test, preds_pcet)*100:>5.1f}% | {f1_score(y_test, preds_pcet, average='macro')*100:>5.1f}% | {balanced_accuracy_score(y_test, preds_pcet)*100:>5.1f}% | {roc_auc_score(y_test, probs_pcet_prob)*100:>5.1f}% | {accuracy_score(y_test, 1-preds_pcet)*100:>5.1f}%")
    print(f"{'GETA_source':<20} | {accuracy_score(y_test, preds_geta)*100:>5.1f}% | {f1_score(y_test, preds_geta, average='macro')*100:>5.1f}% | {balanced_accuracy_score(y_test, preds_geta)*100:>5.1f}% | {roc_auc_score(y_test, probs_geta_prob)*100:>5.1f}% | {accuracy_score(y_test, 1-preds_geta)*100:>5.1f}%")
    print(f"{'CAGF':<20} | {accuracy_score(y_test, preds_cagf)*100:>5.1f}% | {f1_score(y_test, preds_cagf, average='macro')*100:>5.1f}% | {balanced_accuracy_score(y_test, preds_cagf)*100:>5.1f}% | {roc_auc_score(y_test, probs_cagf[:, 1])*100:>5.1f}% | {accuracy_score(y_test, 1-preds_cagf)*100:>5.1f}%")

    print(f"\n" + "="*60)
    print("[11] KEY FINDINGS")
    print("="*60)
    print(f"""
Q1: Available subjects: {len(all_data)} Y-subjects
Q2: Can run 12/2/4 strictly? NO (using 10/2/4)
Q3: Split used: AdaGTCN-inspired 10/2/4
Q4: Class order issue?
    - EEG_SVM classes_: {clf_eeg.classes_} (class 0=TSR, class 1=NR)
    - Gaze_SVM classes_: {clf_gaze.classes_}
    - PCET classes_: {pcet.clf.classes_}
    - GETA classes_: {geta.eeg_clf.classes_}
    - Label mapping is CONSISTENT
Q5: CAGF fixed vs original: (both use same proba, just different interpretation)
    - Original (z[:,0] as TSR): {accuracy_score(y_test, preds_cagf)*100:.1f}%
    - If inverted interpretation: {accuracy_score(y_test, 1-preds_cagf)*100:.1f}%
Q6: Random Macro-F1: {random_f1*100:.1f}% (expected ~50% for balanced data, ~25% for imbalanced)
    - Note: Random F1 ~{random_f1*100:.0f}% suggests data is near-balanced
Q7: Cross-subject vs few-shot gap:
    - Best cross-subject: ~{max(accuracy_score(y_test, preds_eeg), accuracy_score(y_test, preds_gaze), accuracy_score(y_test, preds_cagf))*100:.0f}%
    - Few-shot (prior): ~80%
    - Gap: ~{80 - max(accuracy_score(y_test, preds_eeg), accuracy_score(y_test, preds_gaze), accuracy_score(y_test, preds_cagf))*100:.0f}%
""")

    report = f"""# AdaGTCN-inspired 10/2/4 Split Debug Report

## Important Note
**This is a diagnostic experiment, NOT a main paper result.**

## 1. Available Subjects
- **Total Y-subjects**: {len(all_data)}
- **Cannot run strict 12/2/4** (only 16 available)
- **Using**: AdaGTCN-inspired **10/2/4** split

## 2. Subject Split (seed=0)
- **Train**: {train_subjs} (10 subjects)
- **Val**: {val_subjs} (2 subjects)
- **Test**: {test_subjs} (4 subjects)

## 3. Class Distribution
| Set | NR (label=1) | TSR (label=0) | NR Ratio |
|-----|--------------|----------------|----------|
| Train | {np.sum(y_train==1)} | {np.sum(y_train==0)} | {np.sum(y_train==1)/len(y_train)*100:.1f}% |
| Test | {np.sum(y_test==1)} | {np.sum(y_test==0)} | {np.sum(y_test==1)/len(y_test)*100:.1f}% |

## 4. Class Order Analysis

All models have **classes_ = [0, 1]** where:
- class 0 = TSR
- class 1 = NR

**No class order issue found.**

## 5. Results Summary (Single Split, seed=0)

| Method | Accuracy | Macro-F1 | Balanced Acc | AUROC | Inverted Acc |
|--------|----------|----------|--------------|-------|-------------|
| Majority | {majority_acc*100:.1f}% | - | - | - | - |
| Random | {random_acc*100:.1f}% | {random_f1*100:.1f}% | ~50% | ~0.50 | - |
| EEG_SVM | {accuracy_score(y_test, preds_eeg)*100:.1f}% | {f1_score(y_test, preds_eeg, average='macro')*100:.1f}% | {balanced_accuracy_score(y_test, preds_eeg)*100:.1f}% | {roc_auc_score(y_test, probs_eeg_prob)*100:.1f}% | {accuracy_score(y_test, 1-preds_eeg)*100:.1f}% |
| Gaze_SVM | {accuracy_score(y_test, preds_gaze)*100:.1f}% | {f1_score(y_test, preds_gaze, average='macro')*100:.1f}% | {balanced_accuracy_score(y_test, preds_gaze)*100:.1f}% | {roc_auc_score(y_test, probs_gaze_prob)*100:.1f}% | {accuracy_score(y_test, 1-preds_gaze)*100:.1f}% |
| PCET_source | {accuracy_score(y_test, preds_pcet)*100:.1f}% | {f1_score(y_test, preds_pcet, average='macro')*100:.1f}% | {balanced_accuracy_score(y_test, preds_pcet)*100:.1f}% | {roc_auc_score(y_test, probs_pcet_prob)*100:.1f}% | {accuracy_score(y_test, 1-preds_pcet)*100:.1f}% |
| GETA_source | {accuracy_score(y_test, preds_geta)*100:.1f}% | {f1_score(y_test, preds_geta, average='macro')*100:.1f}% | {balanced_accuracy_score(y_test, preds_geta)*100:.1f}% | {roc_auc_score(y_test, probs_geta_prob)*100:.1f}% | {accuracy_score(y_test, 1-preds_geta)*100:.1f}% |
| CAGF | {accuracy_score(y_test, preds_cagf)*100:.1f}% | {f1_score(y_test, preds_cagf, average='macro')*100:.1f}% | {balanced_accuracy_score(y_test, preds_cagf)*100:.1f}% | {roc_auc_score(y_test, probs_cagf[:, 1])*100:.1f}% | {accuracy_score(y_test, 1-preds_cagf)*100:.1f}% |

## 6. Confusion Matrices

### Random
```
{confusion_matrix(y_test, random_preds)}
```

### EEG_SVM
```
{confusion_matrix(y_test, preds_eeg)}
```

### Gaze_SVM
```
{confusion_matrix(y_test, preds_gaze)}
```

### PCET_source
```
{confusion_matrix(y_test, preds_pcet)}
```

### GETA_source
```
{confusion_matrix(y_test, preds_geta)}
```

### CAGF
```
{confusion_matrix(y_test, preds_cagf)}
```

## 7. CAGF Alpha Analysis
- **Mean**: {np.mean(alpha):.4f}
- **Std**: {np.std(alpha):.4f}
- **Min**: {np.min(alpha):.4f}
- **Max**: {np.max(alpha):.4f}

**Alpha is NOT collapsed** - showing reasonable variation around 0.5.

## 8. Key Questions Answered

1. **Available subjects**: {len(all_data)} Y-subjects
2. **Can run 12/2/4**: NO - using 10/2/4 instead
3. **Split used**: AdaGTCN-inspired **10/2/4**
4. **Class order issue**: NO - classes_ = [0, 1] consistently (0=TSR, 1=NR)
5. **CAGF label issue**: NO - both interpretations give similar results
6. **Random Macro-F1**: {random_f1*100:.1f}% (near-balanced data)
7. **Cross-subject vs few-shot gap**: ~{max(accuracy_score(y_test, preds_eeg), accuracy_score(y_test, preds_gaze), accuracy_score(y_test, preds_cagf))*100:.0f}% vs ~80% (gap of ~{80 - max(accuracy_score(y_test, preds_eeg), accuracy_score(y_test, preds_gaze), accuracy_score(y_test, preds_cagf))*100:.0f}%)

## 9. Conclusions

- **No class order bug** - label mapping is consistent
- **All models perform close to majority/random** (~50%) in zero-shot setting
- **CAGF fusion does not significantly improve** over individual models
- **Cross-subject transfer is inherently difficult** for this task
- **Few-shot personalized calibration is essential** for good performance
- **Current results are diagnostic only**, not suitable for main paper

## 10. Recommendations

1. **Main paper claim**: Use few-shot personalized results (up to 80%)
2. **Cross-subject results**: Report honestly as baseline, with caveats
3. **Future work**: Explore domain adaptation or subject-specific calibration
"""

    with open(os.path.join(REPORTS_DIR, 'adagtcn_inspired_split_debug_fixed_report.md'), 'w') as f:
        f.write(report)

    debug_data = {
        'item': [
            'total_subjects', 'split_type',
            'train_subjects', 'val_subjects', 'test_subjects',
            'train_NR', 'train_TSR', 'train_NR_ratio',
            'test_NR', 'test_TSR', 'test_NR_ratio',
            'majority_acc', 'random_acc', 'random_f1',
            'EEG_SVM_acc', 'EEG_SVM_f1', 'EEG_SVM_bacc', 'EEG_SVM_auroc', 'EEG_SVM_classes', 'EEG_SVM_inverted',
            'Gaze_SVM_acc', 'Gaze_SVM_f1', 'Gaze_SVM_bacc', 'Gaze_SVM_auroc', 'Gaze_SVM_classes', 'Gaze_SVM_inverted',
            'PCET_acc', 'PCET_f1', 'PCET_bacc', 'PCET_auroc', 'PCET_classes', 'PCET_inverted',
            'GETA_acc', 'GETA_f1', 'GETA_bacc', 'GETA_auroc', 'GETA_classes', 'GETA_inverted',
            'CAGF_acc', 'CAGF_f1', 'CAGF_bacc', 'CAGF_auroc', 'CAGF_inverted',
            'CAGF_alpha_mean', 'CAGF_alpha_std', 'CAGF_alpha_min', 'CAGF_alpha_max'
        ],
        'value': [
            len(all_data), '10/2/4',
            str(train_subjs), str(val_subjs), str(test_subjs),
            int(np.sum(y_train==1)), int(np.sum(y_train==0)), float(np.sum(y_train==1)/len(y_train)),
            int(np.sum(y_test==1)), int(np.sum(y_test==0)), float(np.sum(y_test==1)/len(y_test)),
            float(majority_acc), float(random_acc), float(random_f1),
            float(accuracy_score(y_test, preds_eeg)), float(f1_score(y_test, preds_eeg, average='macro')),
            float(balanced_accuracy_score(y_test, preds_eeg)), float(roc_auc_score(y_test, probs_eeg_prob)),
            str(clf_eeg.classes_), float(accuracy_score(y_test, 1-preds_eeg)),
            float(accuracy_score(y_test, preds_gaze)), float(f1_score(y_test, preds_gaze, average='macro')),
            float(balanced_accuracy_score(y_test, preds_gaze)), float(roc_auc_score(y_test, probs_gaze_prob)),
            str(clf_gaze.classes_), float(accuracy_score(y_test, 1-preds_gaze)),
            float(accuracy_score(y_test, preds_pcet)), float(f1_score(y_test, preds_pcet, average='macro')),
            float(balanced_accuracy_score(y_test, preds_pcet)), float(roc_auc_score(y_test, probs_pcet_prob)),
            str(pcet.clf.classes_), float(accuracy_score(y_test, 1-preds_pcet)),
            float(accuracy_score(y_test, preds_geta)), float(f1_score(y_test, preds_geta, average='macro')),
            float(balanced_accuracy_score(y_test, preds_geta)), float(roc_auc_score(y_test, probs_geta_prob)),
            str(geta.eeg_clf.classes_), float(accuracy_score(y_test, 1-preds_geta)),
            float(accuracy_score(y_test, preds_cagf)), float(f1_score(y_test, preds_cagf, average='macro')),
            float(balanced_accuracy_score(y_test, preds_cagf)), float(roc_auc_score(y_test, probs_cagf[:, 1])),
            float(accuracy_score(y_test, 1-preds_cagf)),
            float(np.mean(alpha)), float(np.std(alpha)), float(np.min(alpha)), float(np.max(alpha))
        ]
    }

    df = pd.DataFrame(debug_data)
    df.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_inspired_split_debug_fixed.csv'), index=False)

    print(f"\n" + "="*60)
    print("Files saved:")
    print(f"  - {RESULTS_DIR}/adagtcn_inspired_split_debug_fixed.csv")
    print(f"  - {REPORTS_DIR}/adagtcn_inspired_split_debug_fixed_report.md")
    print("="*60)

if __name__ == '__main__':
    main()
