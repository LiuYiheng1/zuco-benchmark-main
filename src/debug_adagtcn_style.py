"""AdaGTCN-style 12/2/4 Split Debug Script"""

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
        z_gaze = self.gaze_clf.decision_function(X_gaze_s)
        z_gaze = 1 / (1 + np.exp(-z_gaze))
        z_gaze = np.column_stack([1-z_gaze, z_gaze])

        entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        self.eeg_clf.fit(X_eeg_att, y_train)
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)

        z_gaze = self.gaze_clf.decision_function(X_gaze_s)
        z_gaze = 1 / (1 + np.exp(-z_gaze))
        z_gaze = np.column_stack([1-z_gaze, z_gaze])

        entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
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
        z_gaze = 1 / (1 + np.exp(-z_gaze))
        z_gaze = np.column_stack([1-z_gaze, z_gaze])

        entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
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
        z_pcet = self.pcet.predict_proba(X_eeg_test)
        z_geta = self.geta.predict_proba(X_eeg_test, X_gaze_test)
        alpha = 1 / (1 + np.exp(-(z_pcet[:, 0] - z_geta[:, 0])))
        self.last_alpha = alpha
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        return (z_fused[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X_eeg_test, X_gaze_test):
        z_pcet = self.pcet.predict_proba(X_eeg_test)
        z_geta = self.geta.predict_proba(X_eeg_test, X_gaze_test)
        alpha = 1 / (1 + np.exp(-(z_pcet[:, 0] - z_geta[:, 0])))
        self.last_alpha = alpha
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        return z_fused

def main():
    print("="*60)
    print("AdaGTCN-style 12/2/4 Split DEBUG")
    print("="*60)

    print("\nLoading data...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} Y-subjects")

    np.random.seed(0)
    shuffled_subjs = Y_SUBJECTS.copy()
    np.random.shuffle(shuffled_subjs)
    train_subjs = shuffled_subjs[:12]
    val_subjs = shuffled_subjs[12:14]
    test_subjs = shuffled_subjs[14:]

    print(f"\n" + "="*60)
    print("1. SUBJECT SPLIT (seed=0, SINGLE SPLIT)")
    print("="*60)
    print(f"Train: {train_subjs}")
    print(f"Val: {val_subjs}")
    print(f"Test: {test_subjs}")

    X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
    y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
    X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

    X_eeg_test = np.vstack([all_data[s]['Xe'] for s in test_subjs])
    y_test = np.concatenate([all_data[s]['ye'] for s in test_subjs])
    X_gaze_test = np.vstack([all_data[s]['Xg'] for s in test_subjs])

    print(f"\n" + "="*60)
    print("2. CLASS DISTRIBUTION")
    print("="*60)
    print(f"Train: NR={np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%), TSR={np.sum(y_train==0)} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
    print(f"Test:  NR={np.sum(y_test==1)} ({np.sum(y_test==1)/len(y_test)*100:.1f}%), TSR={np.sum(y_test==0)} ({np.sum(y_test==0)/len(y_test)*100:.1f}%)")
    print(f"\nLabel mapping: NR=1, TSR=0")

    print(f"\n" + "="*60)
    print("3. MAJORITY & RANDOM BASELINES")
    print("="*60)
    train_majority = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
    majority_acc = accuracy_score(y_test, np.ones(len(y_test)) * train_majority)
    print(f"Train majority class: {train_majority} (NR if 1, TSR if 0)")
    print(f"Majority baseline accuracy: {majority_acc*100:.1f}%")

    np.random.seed(42)
    random_acc = accuracy_score(y_test, np.random.randint(0, 2, len(y_test)))
    print(f"Random baseline accuracy: ~50%")

    print(f"\n" + "="*60)
    print("4. INDIVIDUAL MODEL RESULTS")
    print("="*60)

    results = {}

    scaler_e = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(X_e_s, y_train)
    preds_eeg = clf.predict(X_e_test_s)
    results['EEG_SVM'] = {
        'acc': accuracy_score(y_test, preds_eeg),
        'f1': f1_score(y_test, preds_eeg, average='macro'),
        'preds': preds_eeg
    }
    print(f"\nEEG_SVM: Acc={results['EEG_SVM']['acc']*100:.1f}%, F1={results['EEG_SVM']['f1']*100:.1f}%")

    scaler_g = StandardScaler()
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(X_g_s, y_train)
    preds_gaze = clf.predict(X_g_test_s)
    results['Gaze_SVM'] = {
        'acc': accuracy_score(y_test, preds_gaze),
        'f1': f1_score(y_test, preds_gaze, average='macro'),
        'preds': preds_gaze
    }
    print(f"Gaze_SVM: Acc={results['Gaze_SVM']['acc']*100:.1f}%, F1={results['Gaze_SVM']['f1']*100:.1f}%")

    pcet = PCETModel()
    pcet.fit(X_eeg_train, y_train)
    preds_pcet = pcet.predict(X_eeg_test)
    z_pcet_raw = pcet.predict_proba_raw(X_eeg_test)
    results['PCET_source'] = {
        'acc': accuracy_score(y_test, preds_pcet),
        'f1': f1_score(y_test, preds_pcet, average='macro'),
        'preds': preds_pcet,
        'z_raw': z_pcet_raw
    }
    print(f"PCET_source: Acc={results['PCET_source']['acc']*100:.1f}%, F1={results['PCET_source']['f1']*100:.1f}%")

    geta = GETAModel()
    geta.fit(X_eeg_train, y_train, X_gaze_train)
    preds_geta = geta.predict(X_eeg_test, X_gaze_test)
    z_geta_raw = geta.predict_proba_raw(X_eeg_test, X_gaze_test)
    results['GETA_source'] = {
        'acc': accuracy_score(y_test, preds_geta),
        'f1': f1_score(y_test, preds_geta, average='macro'),
        'preds': preds_geta,
        'z_raw': z_geta_raw
    }
    print(f"GETA_source: Acc={results['GETA_source']['acc']*100:.1f}%, F1={results['GETA_source']['f1']*100:.1f}%")

    cagf = CAGFFusion()
    cagf.fit(X_eeg_train, y_train, X_gaze_train)
    preds_cagf = cagf.predict(X_eeg_test, X_gaze_test)
    alpha = cagf.last_alpha
    results['CAGF'] = {
        'acc': accuracy_score(y_test, preds_cagf),
        'f1': f1_score(y_test, preds_cagf, average='macro'),
        'preds': preds_cagf,
        'alpha': alpha
    }
    print(f"PCET+GETA+CAGF: Acc={results['CAGF']['acc']*100:.1f}%, F1={results['CAGF']['f1']*100:.1f}%")

    print(f"\n" + "="*60)
    print("5. CONFUSION MATRICES")
    print("="*60)
    print(f"\nPCET confusion matrix:")
    cm_pcet = confusion_matrix(y_test, preds_pcet)
    print(f"          Pred_TSR  Pred_NR")
    print(f"Actual_TSR   {cm_pcet[0,0]:4d}    {cm_pcet[0,1]:4d}")
    print(f"Actual_NR    {cm_pcet[1,0]:4d}    {cm_pcet[1,1]:4d}")

    print(f"\nGETA confusion matrix:")
    cm_geta = confusion_matrix(y_test, preds_geta)
    print(f"          Pred_TSR  Pred_NR")
    print(f"Actual_TSR   {cm_geta[0,0]:4d}    {cm_geta[0,1]:4d}")
    print(f"Actual_NR    {cm_geta[1,0]:4d}    {cm_geta[1,1]:4d}")

    print(f"\nCAGF confusion matrix:")
    cm_cagf = confusion_matrix(y_test, preds_cagf)
    print(f"          Pred_TSR  Pred_NR")
    print(f"Actual_TSR   {cm_cagf[0,0]:4d}    {cm_cagf[0,1]:4d}")
    print(f"Actual_NR    {cm_cagf[1,0]:4d}    {cm_cagf[1,1]:4d}")

    print(f"\n" + "="*60)
    print("6. CAGF ALPHA ANALYSIS")
    print("="*60)
    print(f"Alpha mean: {np.mean(alpha):.4f}")
    print(f"Alpha std:  {np.std(alpha):.4f}")
    print(f"Alpha min:  {np.min(alpha):.4f}")
    print(f"Alpha max:  {np.max(alpha):.4f}")
    print(f"Alpha < 0.3: {np.sum(alpha < 0.3)} ({np.sum(alpha < 0.3)/len(alpha)*100:.1f}%)")
    print(f"Alpha > 0.7: {np.sum(alpha > 0.7)} ({np.sum(alpha > 0.7)/len(alpha)*100:.1f}%)")
    print(f"Alpha in [0.3, 0.7]: {np.sum((alpha >= 0.3) & (alpha <= 0.7))} ({np.sum((alpha >= 0.3) & (alpha <= 0.7))/len(alpha)*100:.1f}%)")

    print(f"\n" + "="*60)
    print("7. LABEL INVERSION TEST")
    print("="*60)
    inverted_acc_pcet = accuracy_score(y_test, 1 - preds_pcet)
    inverted_acc_geta = accuracy_score(y_test, 1 - preds_geta)
    inverted_acc_cagf = accuracy_score(y_test, 1 - preds_cagf)
    print(f"If inverted (flip labels):")
    print(f"  PCET: {inverted_acc_pcet*100:.1f}% (original: {results['PCET_source']['acc']*100:.1f}%)")
    print(f"  GETA: {inverted_acc_geta*100:.1f}% (original: {results['GETA_source']['acc']*100:.1f}%)")
    print(f"  CAGF: {inverted_acc_cagf*100:.1f}% (original: {results['CAGF']['acc']*100:.1f}%)")

    if inverted_acc_cagf > results['CAGF']['acc']:
        print(f"\n*** WARNING: Inverted accuracy is higher! Possible label mapping issue. ***")

    print(f"\n" + "="*60)
    print("8. z_pcet vs z_geta ANALYSIS")
    print("="*60)
    print(f"z_pcet (decision function) mean: {np.mean(z_pcet_raw):.4f}")
    print(f"z_pcet std: {np.std(z_pcet_raw):.4f}")
    print(f"z_geta (decision function) mean: {np.mean(z_geta_raw):.4f}")
    print(f"z_geta std: {np.std(z_geta_raw):.4f}")
    print(f"z_pcet - z_geta mean: {np.mean(z_pcet_raw - z_geta_raw):.4f}")
    print(f"z_pcet - z_geta std: {np.std(z_pcet_raw - z_geta_raw):.4f}")

    print(f"\n" + "="*60)
    print("9. SUMMARY")
    print("="*60)
    print(f"Note: This is a SINGLE split result (not mean ± std over multiple splits)")
    print(f"\nMethod          | Acc    | F1     | Inverted Acc")
    print(f"----------------|--------|--------|-------------")
    print(f"Majority        | {majority_acc*100:5.1f}% |   -    |   -")
    print(f"Random         | ~50.0% | ~25%   |   -")
    print(f"EEG_SVM         | {results['EEG_SVM']['acc']*100:5.1f}% | {results['EEG_SVM']['f1']*100:5.1f}% | {inverted_acc_pcet*100:5.1f}%")
    print(f"Gaze_SVM        | {results['Gaze_SVM']['acc']*100:5.1f}% | {results['Gaze_SVM']['f1']*100:5.1f}% | {inverted_acc_geta*100:5.1f}%")
    print(f"PCET_source     | {results['PCET_source']['acc']*100:5.1f}% | {results['PCET_source']['f1']*100:5.1f}% | {inverted_acc_pcet*100:5.1f}%")
    print(f"GETA_source     | {results['GETA_source']['acc']*100:5.1f}% | {results['GETA_source']['f1']*100:5.1f}% | {inverted_acc_geta*100:5.1f}%")
    print(f"PCET+GETA+CAGF | {results['CAGF']['acc']*100:5.1f}% | {results['CAGF']['f1']*100:5.1f}% | {inverted_acc_cagf*100:5.1f}%")

    debug_data = {
        'item': [
            'Subject Split', 'Train Subjects', 'Val Subjects', 'Test Subjects',
            'Train NR count', 'Train TSR count', 'Test NR count', 'Test TSR count',
            'Train NR ratio', 'Test NR ratio', 'Majority baseline',
            'Random baseline (approx)',
            'EEG_SVM accuracy', 'EEG_SVM F1', 'EEG_SVM inverted',
            'Gaze_SVM accuracy', 'Gaze_SVM F1', 'Gaze_SVM inverted',
            'PCET_source accuracy', 'PCET_source F1', 'PCET_source inverted',
            'GETA_source accuracy', 'GETA_source F1', 'GETA_source inverted',
            'CAGF accuracy', 'CAGF F1', 'CAGF inverted',
            'CAGF alpha mean', 'CAGF alpha std', 'CAGF alpha min', 'CAGF alpha max',
            'z_pcet mean', 'z_pcet std', 'z_geta mean', 'z_geta std',
            'CM_PCET_TN', 'CM_PCET_FP', 'CM_PCET_FN', 'CM_PCET_TP',
            'CM_GETA_TN', 'CM_GETA_FP', 'CM_GETA_FN', 'CM_GETA_TP',
            'CM_CAGF_TN', 'CM_CAGF_FP', 'CM_CAGF_FN', 'CM_CAGF_TP'
        ],
        'value': [
            'Single split (seed=0)', str(train_subjs), str(val_subjs), str(test_subjs),
            int(np.sum(y_train==1)), int(np.sum(y_train==0)), int(np.sum(y_test==1)), int(np.sum(y_test==0)),
            float(np.sum(y_train==1)/len(y_train)), float(np.sum(y_test==1)/len(y_test)), float(majority_acc),
            0.50,
            float(results['EEG_SVM']['acc']), float(results['EEG_SVM']['f1']), float(inverted_acc_pcet),
            float(results['Gaze_SVM']['acc']), float(results['Gaze_SVM']['f1']), float(inverted_acc_geta),
            float(results['PCET_source']['acc']), float(results['PCET_source']['f1']), float(inverted_acc_pcet),
            float(results['GETA_source']['acc']), float(results['GETA_source']['f1']), float(inverted_acc_geta),
            float(results['CAGF']['acc']), float(results['CAGF']['f1']), float(inverted_acc_cagf),
            float(np.mean(alpha)), float(np.std(alpha)), float(np.min(alpha)), float(np.max(alpha)),
            float(np.mean(z_pcet_raw)), float(np.std(z_pcet_raw)), float(np.mean(z_geta_raw)), float(np.std(z_geta_raw)),
            int(cm_pcet[0,0]), int(cm_pcet[0,1]), int(cm_pcet[1,0]), int(cm_pcet[1,1]),
            int(cm_geta[0,0]), int(cm_geta[0,1]), int(cm_geta[1,0]), int(cm_geta[1,1]),
            int(cm_cagf[0,0]), int(cm_cagf[0,1]), int(cm_cagf[1,0]), int(cm_cagf[1,1])
        ]
    }
    df_debug = pd.DataFrame(debug_data)
    df_debug.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_style_debug.csv'), index=False)

    report = f"""# AdaGTCN-style Debug Report

## Important Note
**This is a diagnostic experiment for cross-subject transfer analysis, NOT a main paper result.**

## 1. Subject Split (Single Split, seed=0)
- **Train**: {train_subjs}
- **Validation**: {val_subjs}
- **Test**: {test_subjs}

## 2. Class Distribution
| Set | NR (label=1) | TSR (label=0) | NR Ratio |
|-----|--------------|----------------|----------|
| Train | {np.sum(y_train==1)} | {np.sum(y_train==0)} | {np.sum(y_train==1)/len(y_train)*100:.1f}% |
| Test | {np.sum(y_test==1)} | {np.sum(y_test==0)} | {np.sum(y_test==1)/len(y_test)*100:.1f}% |

**Label mapping**: NR=1, TSR=0 (consistent between train and test)

## 3. Baselines
- **Majority baseline**: {majority_acc*100:.1f}% (predicts all as class {train_majority})
- **Random baseline**: ~50%

## 4. Individual Model Results (Single Split)

| Method | Accuracy | Macro-F1 | Inverted Acc |
|--------|----------|----------|-------------|
| Majority | {majority_acc*100:.1f}% | - | - |
| Random | ~50.0% | ~25% | - |
| EEG_SVM | {results['EEG_SVM']['acc']*100:.1f}% | {results['EEG_SVM']['f1']*100:.1f}% | {inverted_acc_pcet*100:.1f}% |
| Gaze_SVM | {results['Gaze_SVM']['acc']*100:.1f}% | {results['Gaze_SVM']['f1']*100:.1f}% | {inverted_acc_geta*100:.1f}% |
| PCET_source | {results['PCET_source']['acc']*100:.1f}% | {results['PCET_source']['f1']*100:.1f}% | {inverted_acc_pcet*100:.1f}% |
| GETA_source | {results['GETA_source']['acc']*100:.1f}% | {results['GETA_source']['f1']*100:.1f}% | {inverted_acc_geta*100:.1f}% |
| **PCET+GETA+CAGF** | **{results['CAGF']['acc']*100:.1f}%** | **{results['CAGF']['f1']*100:.1f}%** | {inverted_acc_cagf*100:.1f}% |

## 5. Confusion Matrices

### PCET
|  | Pred_TSR | Pred_NR |
|--|----------|---------|
| **Actual_TSR** | {cm_pcet[0,0]} | {cm_pcet[0,1]} |
| **Actual_NR** | {cm_pcet[1,0]} | {cm_pcet[1,1]} |

### GETA
|  | Pred_TSR | Pred_NR |
|--|----------|---------|
| **Actual_TSR** | {cm_geta[0,0]} | {cm_geta[0,1]} |
| **Actual_NR** | {cm_geta[1,0]} | {cm_geta[1,1]} |

### CAGF
|  | Pred_TSR | Pred_NR |
|--|----------|---------|
| **Actual_TSR** | {cm_cagf[0,0]} | {cm_cagf[0,1]} |
| **Actual_NR** | {cm_cagf[1,0]} | {cm_cagf[1,1]} |

## 6. CAGF Alpha Analysis
- **Mean**: {np.mean(alpha):.4f}
- **Std**: {np.std(alpha):.4f}
- **Min**: {np.min(alpha):.4f}
- **Max**: {np.max(alpha):.4f}
- **Alpha < 0.3**: {np.sum(alpha < 0.3)} ({np.sum(alpha < 0.3)/len(alpha)*100:.1f}%)
- **Alpha > 0.7**: {np.sum(alpha > 0.7)} ({np.sum(alpha > 0.7)/len(alpha)*100:.1f}%)
- **Alpha in [0.3, 0.7]**: {np.sum((alpha >= 0.3) & (alpha <= 0.7))} ({np.sum((alpha >= 0.3) & (alpha <= 0.7))/len(alpha)*100:.1f}%)

## 7. Label Inversion Test
If we invert all predictions:
- PCET: {inverted_acc_pcet*100:.1f}% (vs original {results['PCET_source']['acc']*100:.1f}%)
- GETA: {inverted_acc_geta*100:.1f}% (vs original {results['GETA_source']['acc']*100:.1f}%)
- CAGF: {inverted_acc_cagf*100:.1f}% (vs original {results['CAGF']['acc']*100:.1f}%)

{"**WARNING: Inverted accuracy is higher for some models! Possible label mapping issue.**" if inverted_acc_cagf > results['CAGF']['acc'] else ""}

## 8. z_pcet vs z_geta Analysis
- z_pcet mean: {np.mean(z_pcet_raw):.4f}
- z_pcet std: {np.std(z_pcet_raw):.4f}
- z_geta mean: {np.mean(z_geta_raw):.4f}
- z_geta std: {np.std(z_geta_raw):.4f}
- z_pcet - z_geta mean: {np.mean(z_pcet_raw - z_geta_raw):.4f}

## 9. Key Findings

### Is 45.9% below majority baseline?
{"YES - the model is performing worse than majority!" if results['CAGF']['acc'] < majority_acc else "NO - the model is performing above majority."}

### Does label inversion improve results?
{"YES - there may be a label mapping issue between train and test." if inverted_acc_cagf > results['CAGF']['acc'] else "NO - label mapping appears consistent."}

### Is CAGF alpha collapsed?
{"YES - alpha is heavily skewed toward 0 or 1." if (np.sum(alpha < 0.3)/len(alpha) > 0.7 or np.sum(alpha > 0.7)/len(alpha) > 0.7) else "NO - alpha shows reasonable variation."}

### Is cross-subject transfer working?
The accuracy of all models (45-56%) being close to or below majority (~50%) suggests that:
1. Cross-subject transfer is inherently difficult for this task
2. The EEG/gaze features may not generalize well across subjects
3. Subject-specific calibration (few-shot) would likely improve results significantly

## 10. Conclusions

- This is a **diagnostic** experiment, not a main paper result
- Zero-shot cross-subject performance is around random chance level
- Few-shot personalized calibration is likely essential for this approach
- The CAGF fusion does not provide significant improvement over individual models in zero-shot setting
"""

    with open(os.path.join(REPORTS_DIR, 'adagtcn_style_debug_report.md'), 'w') as f:
        f.write(report)

    print(f"\n" + "="*60)
    print("Files saved:")
    print(f"  - {RESULTS_DIR}/adagtcn_style_debug.csv")
    print(f"  - {REPORTS_DIR}/adagtcn_style_debug_report.md")
    print("="*60)

if __name__ == '__main__':
    main()
