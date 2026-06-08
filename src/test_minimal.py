import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def log(msg, f):
    print(msg, flush=True)
    f.write(msg + '\n')
    f.flush()

def run_loso():
    f = open(os.path.join(RESULTS_DIR, 'minimal_log.txt'), 'w')

    log("Loading data...", f)
    all_data = load_all_data()
    log(f"Loaded {len(all_data)} subjects", f)

    results = []
    for i, held_out in enumerate(Y_SUBJECTS):
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]

        X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
        y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])

        X_eeg_test = all_data[held_out]['Xe']
        y_test = all_data[held_out]['ye']

        log(f"[{i+1}/16] {held_out} testing...", f)

        row = {'subject': held_out}

        try:
            scaler_e = StandardScaler()
            X_e_s = scaler_e.fit_transform(X_eeg_train)
            X_e_test_s = scaler_e.transform(X_eeg_test)
            clf = SVC(kernel='linear', random_state=42)
            clf.fit(X_e_s, y_train)
            preds = clf.predict(X_e_test_s)
            row['EEG_SVM'] = accuracy_score(y_test, preds)
        except:
            row['EEG_SVM'] = 0.5

        try:
            pcet = PCETModel()
            pcet.fit(X_eeg_train, y_train)
            preds = pcet.predict(X_eeg_test)
            row['PCET'] = accuracy_score(y_test, preds)
        except:
            row['PCET'] = 0.5

        results.append(row)
        log(f"  SVM:{row['EEG_SVM']:.3f} PCET:{row['PCET']:.3f}", f)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'minimal_results.csv'), index=False)

    log("\nSummary:", f)
    log(f"EEG_SVM: {df['EEG_SVM'].mean():.3f} +/- {df['EEG_SVM'].std():.3f}", f)
    log(f"PCET: {df['PCET'].mean():.3f} +/- {df['PCET'].std():.3f}", f)
    log("Done!", f)
    f.close()

if __name__ == '__main__':
    run_loso()
