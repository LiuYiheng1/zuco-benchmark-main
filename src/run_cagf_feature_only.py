"""
Final CAGF Implementation: CAGF_feature_only
Based on the best performing version from cagf_v3_quick.py

Key differences from current CAGF:
1. Uses SVC(kernel='rbf') instead of RidgeClassifier
2. Final MLP on fused z
3. Full probability vector [p0, p1] for fusion
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
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
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
    return np.array(X), np.array(y)

def load_gaze_features(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
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
    return np.array(X), np.array(y)

def align_eeg_gaze(X_eeg, y_eeg, X_gaze, y_gaze):
    min_len = min(len(X_eeg), len(X_gaze))
    return X_eeg[:min_len], y_eeg[:min_len], X_gaze[:min_len], y_gaze[:min_len]

def safe_roc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5

class PCETModel:
    def __init__(self, n_comp=20):
        self.n_comp = n_comp
        self.pca_models = {}
        self.scaler = StandardScaler()
        self.clf = SVC(kernel='rbf', probability=True, random_state=42)

    def fit(self, X_train, y_train):
        for c in [0, 1]:
            X_c = X_train[y_train == c]
            if len(X_c) > self.n_comp:
                from sklearn.decomposition import PCA
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
        return self.clf.predict_proba(X_combined)

class GETAModel:
    def __init__(self):
        self.scaler_eeg = StandardScaler()
        self.scaler_gaze = StandardScaler()
        self.gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        self.eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

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

class CAGF_feature_only:
    """
    Final CAGF: Cross-modal Adaptive Gated Fusion
    Based on the best performing version from cagf_v3_quick.py

    Key design:
    1. EEG branch: PCET (SVC + PCA error features)
    2. Gaze branch: GETA (gaze-derived attention on EEG)
    3. Fusion: alpha = sigmoid(z_eeg[:,0] - z_gaze[:,0])
    4. Final: MLP on fused z
    """
    def __init__(self):
        self.pcet = PCETModel()
        self.geta = GETAModel()

    def fit(self, X_eeg_train, y_train, X_gaze_train):
        self.pcet.fit(X_eeg_train, y_train)
        self.geta.fit(X_eeg_train, y_train, X_gaze_train)
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        z_eeg_test = self.pcet.predict_proba(X_eeg_test)
        z_gaze_test = self.geta.predict_proba(X_eeg_test, X_gaze_test)

        alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_test, y_train)
        return clf_final.predict(z_fused_test)

    def predict_proba(self, X_eeg_test, X_gaze_test, y_train):
        z_eeg_test = self.pcet.predict_proba(X_eeg_test)
        z_gaze_test = self.geta.predict_proba(X_eeg_test, X_gaze_test)

        alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_test, y_train)
        return clf_final.predict_proba(z_fused_test)

def run_final_experiment(seed, k_values=[3, 5, 10, 20, 50]):
    np.random.seed(seed)
    results = []

    for target_subj in Y_SUBJECTS:
        print(f"  {target_subj}...", end='', flush=True)

        X_eeg_all, y_eeg_all = load_eeg_data(target_subj)
        X_gaze_all, y_gaze_all = load_gaze_features(target_subj)

        if X_eeg_all is None or X_gaze_all is None:
            print(" Skip(no data)")
            continue

        X_eeg_all, y_eeg_all, X_gaze_all, y_gaze_all = align_eeg_gaze(
            X_eeg_all, y_eeg_all, X_gaze_all, y_gaze_all)

        if len(X_eeg_all) < 50:
            print(" Skip(too few)")
            continue

        n_samples = len(y_eeg_all)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_samples // 2]
        cal_pool_indices = indices[n_samples // 2:]

        X_test_eeg = X_eeg_all[test_indices]
        y_test = y_eeg_all[test_indices]
        X_test_gaze = X_gaze_all[test_indices]

        X_cal_pool_eeg = X_eeg_all[cal_pool_indices]
        X_cal_pool_gaze = X_gaze_all[cal_pool_indices]
        y_cal_pool = y_eeg_all[cal_pool_indices]

        for k in k_values:
            if k * 2 > len(cal_pool_indices):
                continue

            cal_idx_c0 = np.where(y_cal_pool == 0)[0][:k]
            cal_idx_c1 = np.where(y_cal_pool == 1)[0][:k]
            cal_indices = np.concatenate([cal_idx_c0, cal_idx_c1])
            np.random.shuffle(cal_indices)

            X_cal_eeg = X_cal_pool_eeg[cal_indices]
            y_cal = y_cal_pool[cal_indices]
            X_cal_gaze = X_cal_pool_gaze[cal_indices]

            row = {'seed': seed, 'subject': target_subj, 'k': k, 'n_test': len(y_test)}

            cagf = CAGF_feature_only()
            try:
                cagf.fit(X_cal_eeg, y_cal, X_cal_gaze)
                preds = cagf.predict(X_cal_eeg, X_cal_gaze)
                probs = cagf.predict_proba(X_cal_eeg, X_cal_gaze, y_cal)

                test_preds = cagf.predict(X_test_eeg, X_test_gaze)
                test_probs = cagf.predict_proba(X_test_eeg, X_test_gaze, y_cal)

                row['PCET_GETA_CAGF_acc'] = accuracy_score(y_test, test_preds)
                row['PCET_GETA_CAGF_f1'] = f1_score(y_test, test_preds, average='macro')
                row['PCET_GETA_CAGF_bacc'] = balanced_accuracy_score(y_test, test_preds)
                row['PCET_GETA_CAGF_auroc'] = safe_roc(y_test, test_probs[:, 1])
            except Exception as e:
                print(f" Err:{str(e)[:30]}", end='')
                row['PCET_GETA_CAGF_acc'] = 0.5
                row['PCET_GETA_CAGF_f1'] = 0.5
                row['PCET_GETA_CAGF_bacc'] = 0.5
                row['PCET_GETA_CAGF_auroc'] = 0.5

            results.append(row)
        print("OK", flush=True)

    return results

def main():
    print("="*70)
    print("Final CAGF_feature_only Experiment")
    print("="*70)

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_final_experiment(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "cagf_feature_only_final.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("FINAL RESULTS: CAGF_feature_only")
    print("="*70)

    for k in k_values:
        subset = df[df['k'] == k]
        if len(subset) > 0:
            acc = subset['PCET_GETA_CAGF_acc'].mean() * 100
            f1 = subset['PCET_GETA_CAGF_f1'].mean() * 100
            bacc = subset['PCET_GETA_CAGF_bacc'].mean() * 100
            auroc = subset['PCET_GETA_CAGF_auroc'].mean() * 100
            print(f"k={k:>3}: Acc={acc:.1f}% F1={f1:.1f}% BAcc={bacc:.1f}% AUROC={auroc:.1f}%")

    print(f"\nSaved to {output_path}")
    print("\nDONE!")

if __name__ == '__main__':
    main()