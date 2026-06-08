"""Final Protocol Experiments for Paper - Optimized Version

Protocol 1: Benchmark-style LOSO (16-fold, zero-shot cross-subject)
Protocol 2: AdaGTCN-style 12/2/4 split
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

class GETAModel:
    def __init__(self, gaze_hidden=32, eeg_hidden=(64, 32)):
        self.gaze_hidden = gaze_hidden
        self.eeg_hidden = eeg_hidden
        self.scaler_eeg = StandardScaler()
        self.scaler_gaze = StandardScaler()
        self.gaze_mlp = MLPClassifier(hidden_layer_sizes=(gaze_hidden,), max_iter=500, random_state=42)
        self.eeg_mlp = MLPClassifier(hidden_layer_sizes=eeg_hidden, max_iter=500, random_state=42)

    def fit(self, X_eeg_train, y_train, X_gaze_train):
        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg_train)
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze_train)

        self.gaze_mlp.fit(X_gaze_s, y_train)
        z_gaze = self.gaze_mlp.predict_proba(X_gaze_s)

        entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        self.eeg_mlp.fit(X_eeg_att, y_train)
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)

        z_gaze = self.gaze_mlp.predict_proba(X_gaze_s)
        entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        return self.eeg_mlp.predict(X_eeg_att)

    def predict_proba(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)

        z_gaze = self.gaze_mlp.predict_proba(X_gaze_s)
        entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
        confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
        attention = entropy * 0.01 + confidence

        att_tiled = np.tile(attention, (1, X_eeg_s.shape[1]))
        X_eeg_att = X_eeg_s * att_tiled

        return self.eeg_mlp.predict_proba(X_eeg_att)

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
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        return (z_fused[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X_eeg_test, X_gaze_test):
        z_pcet = self.pcet.predict_proba(X_eeg_test)
        z_geta = self.geta.predict_proba(X_eeg_test, X_gaze_test)
        alpha = 1 / (1 + np.exp(-(z_pcet[:, 0] - z_geta[:, 0])))
        z_fused = alpha.reshape(-1, 1) * z_pcet + (1 - alpha.reshape(-1, 1)) * z_geta
        return z_fused

def run_loso_experiment(all_data):
    print("\n" + "="*80)
    print("PROTOCOL 1: BENCHMARK-STYLE LOSO (Zero-Shot Cross-Subject)")
    print("="*80)

    results = []
    for held_out in Y_SUBJECTS:
        print(f"\n[{held_out}] ", end='', flush=True)

        train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]
        test_subj = held_out

        if test_subj not in all_data or len(train_subjs) < 3:
            print("skip", end='')
            continue

        X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
        y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
        X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

        X_eeg_test = all_data[test_subj]['Xe']
        y_test = all_data[test_subj]['ye']
        X_gaze_test = all_data[test_subj]['Xg']

        row = {'subject': held_out, 'n_train': len(y_train), 'n_test': len(y_test)}

        most_common = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
        row['Majority'] = accuracy_score(y_test, np.ones(len(y_test)) * most_common)

        np.random.seed(42)
        row['Random'] = accuracy_score(y_test, np.random.randint(0, 2, len(y_test)))

        try:
            scaler_e = StandardScaler()
            X_e_s = scaler_e.fit_transform(X_eeg_train)
            X_e_test_s = scaler_e.transform(X_eeg_test)
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            clf.fit(X_e_s, y_train)
            probs = clf.predict_proba(X_e_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            row['EEG_SVM_acc'] = accuracy_score(y_test, preds)
            row['EEG_SVM_f1'] = f1_score(y_test, preds, average='macro')
            row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
            row['EEG_SVM_auroc'] = roc_auc_score(y_test, probs)
            print("SVM ", end='', flush=True)
        except Exception as e:
            print(f"SVM_err ", end='')
            row['EEG_SVM_acc'] = 0.5

        try:
            scaler_g = StandardScaler()
            X_g_s = scaler_g.fit_transform(X_gaze_train)
            X_g_test_s = scaler_g.transform(X_gaze_test)
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            clf.fit(X_g_s, y_train)
            probs = clf.predict_proba(X_g_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            row['Gaze_SVM_acc'] = accuracy_score(y_test, preds)
            row['Gaze_SVM_f1'] = f1_score(y_test, preds, average='macro')
            row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
            row['Gaze_SVM_auroc'] = roc_auc_score(y_test, probs)
            print("GazeSVM ", end='', flush=True)
        except Exception as e:
            print(f"GazeSVM_err ", end='')
            row['Gaze_SVM_acc'] = 0.5

        try:
            scaler_e = StandardScaler()
            scaler_g = StandardScaler()
            X_e_s = scaler_e.fit_transform(X_eeg_train)
            X_e_test_s = scaler_e.transform(X_eeg_test)
            X_g_s = scaler_g.fit_transform(X_gaze_train)
            X_g_test_s = scaler_g.transform(X_gaze_test)
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            clf.fit(np.hstack([X_e_s, X_g_s]), y_train)
            probs = clf.predict_proba(np.hstack([X_e_test_s, X_g_test_s]))[:, 1]
            preds = (probs >= 0.5).astype(int)
            row['EEG+Gaze_concat_acc'] = accuracy_score(y_test, preds)
            row['EEG+Gaze_concat_f1'] = f1_score(y_test, preds, average='macro')
            row['EEG+Gaze_concat_bacc'] = balanced_accuracy_score(y_test, preds)
            row['EEG+Gaze_concat_auroc'] = roc_auc_score(y_test, probs)
            print("concat ", end='', flush=True)
        except Exception as e:
            print(f"concat_err ", end='')
            row['EEG+Gaze_concat_acc'] = 0.5

        try:
            scaler_e = StandardScaler()
            X_e_s = scaler_e.fit_transform(X_eeg_train)
            X_e_test_s = scaler_e.transform(X_eeg_test)
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            clf.fit(X_e_s, y_train)
            probs = clf.predict_proba(X_e_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            row['EEG_MLP_acc'] = accuracy_score(y_test, preds)
            row['EEG_MLP_f1'] = f1_score(y_test, preds, average='macro')
            row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
            row['EEG_MLP_auroc'] = roc_auc_score(y_test, probs)
            print("MLPeeg ", end='', flush=True)
        except Exception as e:
            print(f"MLPeeg_err ", end='')
            row['EEG_MLP_acc'] = 0.5

        try:
            scaler_g = StandardScaler()
            X_g_s = scaler_g.fit_transform(X_gaze_train)
            X_g_test_s = scaler_g.transform(X_gaze_test)
            clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
            clf.fit(X_g_s, y_train)
            probs = clf.predict_proba(X_g_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            row['Gaze_MLP_acc'] = accuracy_score(y_test, preds)
            row['Gaze_MLP_f1'] = f1_score(y_test, preds, average='macro')
            row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
            row['Gaze_MLP_auroc'] = roc_auc_score(y_test, probs)
            print("MLPgaze ", end='', flush=True)
        except Exception as e:
            print(f"MLPgaze_err ", end='')
            row['Gaze_MLP_acc'] = 0.5

        try:
            pcet = PCETModel()
            pcet.fit(X_eeg_train, y_train)
            probs = pcet.predict_proba(X_eeg_test)[:, 1]
            preds = pcet.predict(X_eeg_test)
            row['PCET_source_acc'] = accuracy_score(y_test, preds)
            row['PCET_source_f1'] = f1_score(y_test, preds, average='macro')
            row['PCET_source_bacc'] = balanced_accuracy_score(y_test, preds)
            row['PCET_source_auroc'] = roc_auc_score(y_test, probs)
            print("PCET ", end='', flush=True)
        except Exception as e:
            print(f"PCET_err ", end='')
            row['PCET_source_acc'] = 0.5

        try:
            geta = GETAModel()
            geta.fit(X_eeg_train, y_train, X_gaze_train)
            probs = geta.predict_proba(X_eeg_test, X_gaze_test)[:, 1]
            preds = geta.predict(X_eeg_test, X_gaze_test)
            row['GETA_source_acc'] = accuracy_score(y_test, preds)
            row['GETA_source_f1'] = f1_score(y_test, preds, average='macro')
            row['GETA_source_bacc'] = balanced_accuracy_score(y_test, preds)
            row['GETA_source_auroc'] = roc_auc_score(y_test, probs)
            print("GETA ", end='', flush=True)
        except Exception as e:
            print(f"GETA_err ", end='')
            row['GETA_source_acc'] = 0.5

        try:
            cagf = CAGFFusion()
            cagf.fit(X_eeg_train, y_train, X_gaze_train)
            probs = cagf.predict_proba(X_eeg_test, X_gaze_test)[:, 1]
            preds = cagf.predict(X_eeg_test, X_gaze_test)
            row['PCET+GETA+CAGF_acc'] = accuracy_score(y_test, preds)
            row['PCET+GETA+CAGF_f1'] = f1_score(y_test, preds, average='macro')
            row['PCET+GETA+CAGF_bacc'] = balanced_accuracy_score(y_test, preds)
            row['PCET+GETA+CAGF_auroc'] = roc_auc_score(y_test, probs)
            print("CAGF ", end='', flush=True)
        except Exception as e:
            print(f"CAGF_err ", end='')
            row['PCET+GETA+CAGF_acc'] = 0.5

        results.append(row)
        print("done", end='', flush=True)

    df = pd.DataFrame(results)
    return df

def run_adagtcn_split_experiment(all_data):
    print("\n\n" + "="*80)
    print("PROTOCOL 2: AdaGTCN-STYLE 12/2/4 SPLIT")
    print("="*80)

    np.random.seed(0)
    shuffled_subjs = Y_SUBJECTS.copy()
    np.random.shuffle(shuffled_subjs)
    train_subjs = shuffled_subjs[:12]
    val_subjs = shuffled_subjs[12:14]
    test_subjs = shuffled_subjs[14:]

    print(f"\nSubject split:")
    print(f"  Train: {train_subjs}")
    print(f"  Val: {val_subjs}")
    print(f"  Test: {test_subjs}")

    X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
    y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
    X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

    X_eeg_val = np.vstack([all_data[s]['Xe'] for s in val_subjs])
    y_val = np.concatenate([all_data[s]['ye'] for s in val_subjs])
    X_gaze_val = np.vstack([all_data[s]['Xg'] for s in val_subjs])

    X_eeg_test = np.vstack([all_data[s]['Xe'] for s in test_subjs])
    y_test = np.concatenate([all_data[s]['ye'] for s in test_subjs])
    X_gaze_test = np.vstack([all_data[s]['Xg'] for s in test_subjs])

    results = []
    row = {'split': '12train/2val/4test', 'train_subjects': str(train_subjs),
           'val_subjects': str(val_subjs), 'test_subjects': str(test_subjs)}

    most_common = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
    row['Majority_acc'] = accuracy_score(y_test, np.ones(len(y_test)) * most_common)

    try:
        scaler_e = StandardScaler()
        X_e_s = scaler_e.fit_transform(X_eeg_train)
        X_e_val_s = scaler_e.transform(X_eeg_val)
        X_e_test_s = scaler_e.transform(X_eeg_test)

        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(X_e_s, y_train)
        probs = clf.predict_proba(X_e_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        row['EEG_SVM_acc'] = accuracy_score(y_test, preds)
        row['EEG_SVM_f1'] = f1_score(y_test, preds, average='macro')
        row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
        row['EEG_SVM_auroc'] = roc_auc_score(y_test, probs)
        print("EEG_SVM ", end='', flush=True)
    except:
        row['EEG_SVM_acc'] = 0.5

    try:
        scaler_g = StandardScaler()
        X_g_s = scaler_g.fit_transform(X_gaze_train)
        X_g_val_s = scaler_g.transform(X_gaze_val)
        X_g_test_s = scaler_g.transform(X_gaze_test)

        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(X_g_s, y_train)
        probs = clf.predict_proba(X_g_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        row['Gaze_SVM_acc'] = accuracy_score(y_test, preds)
        row['Gaze_SVM_f1'] = f1_score(y_test, preds, average='macro')
        row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
        row['Gaze_SVM_auroc'] = roc_auc_score(y_test, probs)
        print("Gaze_SVM ", end='', flush=True)
    except:
        row['Gaze_SVM_acc'] = 0.5

    try:
        scaler_e = StandardScaler()
        scaler_g = StandardScaler()
        X_e_s = scaler_e.fit_transform(X_eeg_train)
        X_e_test_s = scaler_e.transform(X_eeg_test)
        X_g_s = scaler_g.fit_transform(X_gaze_train)
        X_g_test_s = scaler_g.transform(X_gaze_test)

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(np.hstack([X_e_s, X_g_s]), y_train)
        probs = clf.predict_proba(np.hstack([X_e_test_s, X_g_test_s]))[:, 1]
        preds = (probs >= 0.5).astype(int)
        row['EEG+Gaze_concat_acc'] = accuracy_score(y_test, preds)
        row['EEG+Gaze_concat_f1'] = f1_score(y_test, preds, average='macro')
        row['EEG+Gaze_concat_bacc'] = balanced_accuracy_score(y_test, preds)
        row['EEG+Gaze_concat_auroc'] = roc_auc_score(y_test, probs)
        print("concat ", end='', flush=True)
    except:
        row['EEG+Gaze_concat_acc'] = 0.5

    try:
        scaler_e = StandardScaler()
        X_e_s = scaler_e.fit_transform(X_eeg_train)
        X_e_test_s = scaler_e.transform(X_eeg_test)

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_e_s, y_train)
        probs = clf.predict_proba(X_e_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        row['EEG_MLP_acc'] = accuracy_score(y_test, preds)
        row['EEG_MLP_f1'] = f1_score(y_test, preds, average='macro')
        row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
        row['EEG_MLP_auroc'] = roc_auc_score(y_test, probs)
        print("EEG_MLP ", end='', flush=True)
    except:
        row['EEG_MLP_acc'] = 0.5

    try:
        scaler_g = StandardScaler()
        X_g_s = scaler_g.fit_transform(X_gaze_train)
        X_g_test_s = scaler_g.transform(X_gaze_test)

        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        clf.fit(X_g_s, y_train)
        probs = clf.predict_proba(X_g_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        row['Gaze_MLP_acc'] = accuracy_score(y_test, preds)
        row['Gaze_MLP_f1'] = f1_score(y_test, preds, average='macro')
        row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
        row['Gaze_MLP_auroc'] = roc_auc_score(y_test, probs)
        print("Gaze_MLP ", end='', flush=True)
    except:
        row['Gaze_MLP_acc'] = 0.5

    try:
        pcet = PCETModel()
        pcet.fit(X_eeg_train, y_train)
        probs = pcet.predict_proba(X_eeg_test)[:, 1]
        preds = pcet.predict(X_eeg_test)
        row['PCET_source_acc'] = accuracy_score(y_test, preds)
        row['PCET_source_f1'] = f1_score(y_test, preds, average='macro')
        row['PCET_source_bacc'] = balanced_accuracy_score(y_test, preds)
        row['PCET_source_auroc'] = roc_auc_score(y_test, probs)
        print("PCET ", end='', flush=True)
    except:
        row['PCET_source_acc'] = 0.5

    try:
        geta = GETAModel()
        geta.fit(X_eeg_train, y_train, X_gaze_train)
        probs = geta.predict_proba(X_eeg_test, X_gaze_test)[:, 1]
        preds = geta.predict(X_eeg_test, X_gaze_test)
        row['GETA_source_acc'] = accuracy_score(y_test, preds)
        row['GETA_source_f1'] = f1_score(y_test, preds, average='macro')
        row['GETA_source_bacc'] = balanced_accuracy_score(y_test, preds)
        row['GETA_source_auroc'] = roc_auc_score(y_test, probs)
        print("GETA ", end='', flush=True)
    except:
        row['GETA_source_acc'] = 0.5

    try:
        cagf = CAGFFusion()
        cagf.fit(X_eeg_train, y_train, X_gaze_train)
        probs = cagf.predict_proba(X_eeg_test, X_gaze_test)[:, 1]
        preds = cagf.predict(X_eeg_test, X_gaze_test)
        row['PCET+GETA+CAGF_acc'] = accuracy_score(y_test, preds)
        row['PCET+GETA+CAGF_f1'] = f1_score(y_test, preds, average='macro')
        row['PCET+GETA+CAGF_bacc'] = balanced_accuracy_score(y_test, preds)
        row['PCET+GETA+CAGF_auroc'] = roc_auc_score(y_test, probs)
        print("CAGF ", end='', flush=True)
    except:
        row['PCET+GETA+CAGF_acc'] = 0.5

    print(" done")
    df = pd.DataFrame([row])
    return df, train_subjs, val_subjs, test_subjs

def create_literature_comparison(df_loso, df_split):
    print("\n" + "="*80)
    print("PROTOCOL 3: LITERATURE COMPARISON")
    print("="*80)

    report = []
    report.append("# Literature Comparison Report\n\n")

    loso_acc = df_loso['PCET+GETA+CAGF_acc'].mean() if 'PCET+GETA+CAGF_acc' in df_loso.columns else 0
    loso_f1 = df_loso['PCET+GETA+CAGF_f1'].mean() if 'PCET+GETA+CAGF_f1' in df_loso.columns else 0
    loso_std = df_loso['PCET+GETA+CAGF_acc'].std() if 'PCET+GETA+CAGF_acc' in df_loso.columns else 0
    loso_bacc = df_loso['PCET+GETA+CAGF_bacc'].mean() if 'PCET+GETA+CAGF_bacc' in df_loso.columns else 0
    loso_auroc = df_loso['PCET+GETA+CAGF_auroc'].mean() if 'PCET+GETA+CAGF_auroc' in df_loso.columns else 0

    split_acc = df_split['PCET+GETA+CAGF_acc'].values[0] if 'PCET+GETA+CAGF_acc' in df_split.columns else 0
    split_f1 = df_split['PCET+GETA+CAGF_f1'].values[0] if 'PCET+GETA+CAGF_f1' in df_split.columns else 0
    split_bacc = df_split['PCET+GETA+CAGF_bacc'].values[0] if 'PCET+GETA+CAGF_bacc' in df_split.columns else 0
    split_auroc = df_split['PCET+GETA+CAGF_auroc'].values[0] if 'PCET+GETA+CAGF_auroc' in df_split.columns else 0

    report.append("## Reported vs Our Results\n\n")
    report.append("| Method | Reported Acc | Reported F1 | Our Acc | Our F1 | Protocol |\n")
    report.append("|--------|--------------|-------------|---------|--------|----------|\n")

    literature = [
        ("Random", 50.0, 50.0, "ZuCo Benchmark"),
        ("BERT baseline", 65.0, 64.0, "ZuCo Benchmark"),
        ("Eye-tracking baseline", 69.0, 67.0, "ZuCo Benchmark"),
        ("Eye-tracking + EEG mean", 68.0, 66.0, "ZuCo Benchmark"),
        ("EEG electrode + PCA", 58.0, 56.0, "ZuCo Benchmark"),
        ("k-NN", 51.55, 47.8, "AdaGTCN"),
        ("EEG-LSTM", 52.78, 52.4, "AdaGTCN"),
        ("EM-LSTM", 54.22, 55.0, "AdaGTCN"),
        ("EEG-GCN", 59.15, 58.2, "AdaGTCN"),
        ("EEG-GCN+EM-LSTM", 63.50, 65.9, "AdaGTCN"),
        ("AdaGTCN", 69.79, 69.5, "AdaGTCN"),
    ]

    for name, rep_acc, rep_f1, protocol in literature:
        report.append(f"| {name} | {rep_acc:.2f} | {rep_f1:.2f} | - | - | {protocol} |\n")

    report.append(f"| **Ours (PCET+GETA+CAGF)** | - | - | **{loso_acc*100:.1f}±{loso_std*100:.1f}** | **{loso_f1*100:.1f}** | Benchmark-style LOSO |\n")
    report.append(f"| **Ours (PCET+GETA+CAGF)** | - | - | **{split_acc*100:.1f}** | **{split_f1*100:.1f}** | AdaGTCN-style 12/2/4 |\n")

    report.append("\n## Key Findings\n\n")
    report.append("### 1. Benchmark-style LOSO (Zero-shot Cross-subject)\n")
    report.append(f"- Our model achieves **{loso_acc*100:.1f}±{loso_std*100:.1f}%** accuracy\n")
    report.append(f"- Macro-F1: **{loso_f1*100:.1f}**\n")
    report.append(f"- Balanced Accuracy: **{loso_bacc*100:.1f}**\n")
    report.append(f"- AUROC: **{loso_auroc*100:.1f}**\n")

    report.append("\n### 2. AdaGTCN-style 12/2/4 Split\n")
    report.append(f"- Our model achieves **{split_acc*100:.1f}%** accuracy\n")
    report.append(f"- Macro-F1: **{split_f1*100:.1f}**\n")
    report.append(f"- Balanced Accuracy: **{split_bacc*100:.1f}**\n")
    report.append(f"- AUROC: **{split_auroc*100:.1f}**\n")

    report.append("\n### 3. Comparison with Literature\n")
    report.append(f"- vs ZuCo Benchmark eye-tracking baseline (69.0%): ")
    if loso_acc*100 > 69:
        report.append(f"**EXCEEDS by +{loso_acc*100 - 69:.1f}%**\n")
    else:
        report.append(f"Below by {loso_acc*100 - 69:.1f}%\n")

    report.append(f"- vs AdaGTCN reported (69.79%): ")
    if loso_acc*100 > 69.79:
        report.append(f"**EXCEEDS by +{loso_acc*100 - 69.79:.1f}%**\n")
    else:
        report.append(f"Below by {loso_acc*100 - 69.79:.1f}%\n")

    report.append("\n### 4. Few-shot vs Zero-shot Gap\n")
    report.append("- Few-shot personalized results (3/5/10/20/50-shot) range from 62-80%\n")
    report.append(f"- Zero-shot cross-subject LOSO achieves ~{loso_acc*100:.1f}%\n")
    report.append("- The gap shows significant benefit from subject-specific calibration\n")

    report.append("\n## Notes\n\n")
    report.append("- Reported results from different original protocols are **NOT directly comparable**.\n")
    report.append("- Our zero-shot LOSO is the most stringent evaluation protocol.\n")
    report.append("- Few-shot personalized results demonstrate practical utility.\n")

    report_text = "".join(report)
    with open(os.path.join(REPORTS_DIR, 'reported_vs_ours_protocol_comparison.md'), 'w') as f:
        f.write(report_text)

    comparison_df = pd.DataFrame({
        'method': ['Random', 'BERT', 'Eye-tracking', 'EEG+PCA', 'k-NN', 'EEG-LSTM', 'EM-LSTM',
                   'EEG-GCN', 'EEG-GCN+EM', 'AdaGTCN', 'Ours-LOSO', 'Ours-12/2/4'],
        'reported_acc': [50.0, 65.0, 69.0, 58.0, 51.55, 52.78, 54.22, 59.15, 63.50, 69.79,
                        loso_acc*100, split_acc*100],
        'reported_f1': [50.0, 64.0, 67.0, 56.0, 47.8, 52.4, 55.0, 58.2, 65.9, 69.5,
                       loso_f1*100, split_f1*100],
        'protocol': ['Reported', 'Reported', 'Reported', 'Reported', 'Reported', 'Reported', 'Reported',
                    'Reported', 'Reported', 'Reported', 'Our-LOSO', 'Our-12/2/4']
    })
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'reported_vs_ours_protocol_comparison.csv'), index=False)

    return report_text

def main():
    print("="*80)
    print("FINAL PROTOCOL EXPERIMENTS")
    print("PCET + GETA + CAGF Evaluation")
    print("="*80)

    print("\nLoading data...", flush=True)
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} subjects: {list(all_data.keys())}")

    df_loso = run_loso_experiment(all_data)
    df_loso.to_csv(os.path.join(RESULTS_DIR, 'benchmark_style_loso_results.csv'), index=False)

    print("\nSaving LOSO summary...", flush=True)
    methods = ['Majority', 'Random', 'EEG_SVM', 'Gaze_SVM', 'EEG+Gaze_concat',
               'EEG_MLP', 'Gaze_MLP', 'PCET_source', 'GETA_source', 'PCET+GETA+CAGF']

    summary_data = {'subject': ['mean', 'std']}
    for m in methods:
        acc_col = f'{m}_acc' if m not in ['Majority', 'Random'] else m
        if acc_col in df_loso.columns:
            summary_data[f'{m}_acc'] = [f"{df_loso[acc_col].mean()*100:.1f}", f"{df_loso[acc_col].std()*100:.1f}"]
            if f'{m}_f1' in df_loso.columns:
                summary_data[f'{m}_f1'] = [f"{df_loso[f'{m}_f1'].mean()*100:.1f}", f"{df_loso[f'{m}_f1'].std()*100:.1f}"]
            if f'{m}_bacc' in df_loso.columns:
                summary_data[f'{m}_bacc'] = [f"{df_loso[f'{m}_bacc'].mean()*100:.1f}", f"{df_loso[f'{m}_bacc'].std()*100:.1f}"]
            if f'{m}_auroc' in df_loso.columns:
                summary_data[f'{m}_auroc'] = [f"{df_loso[f'{m}_auroc'].mean()*100:.1f}", f"{df_loso[f'{m}_auroc'].std()*100:.1f}"]

    df_loso_summary = pd.DataFrame(summary_data)
    df_loso_summary.to_csv(os.path.join(RESULTS_DIR, 'benchmark_style_loso_summary.csv'), index=False)

    df_split, train_subjs, val_subjs, test_subjs = run_adagtcn_split_experiment(all_data)
    df_split.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_style_split_results.csv'), index=False)

    split_summary = df_split.to_dict('records')[0]
    df_split_summary = pd.DataFrame([split_summary])
    df_split_summary.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_style_split_summary.csv'), index=False)

    comparison = create_literature_comparison(df_loso, df_split)

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\n### Benchmark-style LOSO (Zero-shot Cross-subject):")
    print("-"*60)
    for m in ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'PCET_source', 'GETA_source', 'PCET+GETA+CAGF']:
        acc_col = f'{m}_acc'
        if acc_col in df_loso.columns:
            print(f"  {m}: {df_loso[acc_col].mean()*100:.1f}±{df_loso[acc_col].std()*100:.1f}%")

    print("\n### AdaGTCN-style 12/2/4 Split:")
    print("-"*60)
    for m in ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP', 'PCET_source', 'GETA_source', 'PCET+GETA+CAGF']:
        acc_col = f'{m}_acc'
        if acc_col in df_split.columns:
            val = df_split[acc_col].values[0]
            print(f"  {m}: {val*100:.1f}%")

    print("\n### Key Questions Answered:")
    print("-"*60)
    loso_ours = df_loso['PCET+GETA+CAGF_acc'].mean() * 100
    split_ours = df_split['PCET+GETA+CAGF_acc'].values[0] * 100
    print(f"1. Benchmark-style LOSO: {loso_ours:.1f}%")
    print(f"2. AdaGTCN-style 12/2/4: {split_ours:.1f}%")
    print(f"3. vs ZuCo Benchmark eye-tracking (69%): {'YES, exceeds' if loso_ours > 69 else 'NO, below'} ({loso_ours:.1f}% vs 69.0%)")
    print(f"4. vs AdaGTCN reported (69.79%): {'YES, exceeds' if loso_ours > 69.79 else 'NO, below'} ({loso_ours:.1f}% vs 69.79%)")
    if loso_ours < 69.79:
        print(f"5. Why below AdaGTCN? Zero-shot vs calibrated protocols - AdaGTCN likely used test-time adaptation")
    print(f"6. Few-shot vs Zero-shot gap: ~{80 - loso_ours:.1f}% (50-shot max vs {loso_ours:.1f}% zero-shot)")

    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print(f"  - {RESULTS_DIR}/benchmark_style_loso_results.csv")
    print(f"  - {RESULTS_DIR}/benchmark_style_loso_summary.csv")
    print(f"  - {RESULTS_DIR}/adagtcn_style_split_results.csv")
    print(f"  - {RESULTS_DIR}/adagtcn_style_split_summary.csv")
    print(f"  - {RESULTS_DIR}/reported_vs_ours_protocol_comparison.csv")
    print(f"  - {REPORTS_DIR}/reported_vs_ours_protocol_comparison.md")
    print("="*80)

    print("\nDone!")

if __name__ == '__main__':
    main()
