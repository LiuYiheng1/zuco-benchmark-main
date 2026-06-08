"""
Few-Shot AdaGTCN-Proxy Comparison - Simplified Version
Run first with simpler models to verify protocol works
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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

        z_pcet_full = np.column_stack([1-z_pcet_prob, z_pcet_prob])
        z_geta_full = np.column_stack([1-z_geta_prob, z_geta_prob])

        z_fused = alpha.reshape(-1, 1) * z_pcet_full + (1 - alpha.reshape(-1, 1)) * z_geta_full
        return (z_fused[:, 1] >= 0.5).astype(int), z_fused

def safe_roc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5

def run_few_shot_experiment(seed, k_values=[3, 5, 10, 20, 50]):
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

            scaler_eeg = StandardScaler()
            X_cal_eeg_s = scaler_eeg.fit_transform(X_cal_eeg)
            X_test_eeg_s = scaler_eeg.transform(X_test_eeg)

            scaler_gaze = StandardScaler()
            X_cal_gaze_s = scaler_gaze.fit_transform(X_cal_gaze)
            X_test_gaze_s = scaler_gaze.transform(X_test_gaze)

            row = {
                'seed': seed, 'subject': target_subj, 'k': k,
                'n_test': len(y_test), 'n_cal': len(y_cal),
            }

            clf = RidgeClassifier(alpha=0.1)
            clf.fit(X_cal_eeg_s, y_cal)
            pred = clf.predict(X_test_eeg_s)
            prob = 1 / (1 + np.exp(-clf.decision_function(X_test_eeg_s)))
            row['EEG_SVM_acc'] = accuracy_score(y_test, pred)
            row['EEG_SVM_f1'] = f1_score(y_test, pred, average='macro')
            row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, pred)
            row['EEG_SVM_auroc'] = safe_roc(y_test, prob)

            clf = RidgeClassifier(alpha=0.1)
            clf.fit(X_cal_gaze_s, y_cal)
            pred = clf.predict(X_test_gaze_s)
            prob = 1 / (1 + np.exp(-clf.decision_function(X_test_gaze_s)))
            row['Gaze_SVM_acc'] = accuracy_score(y_test, pred)
            row['Gaze_SVM_f1'] = f1_score(y_test, pred, average='macro')
            row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, pred)
            row['Gaze_SVM_auroc'] = safe_roc(y_test, prob)

            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
            clf.fit(X_cal_eeg_s, y_cal)
            pred = clf.predict(X_test_eeg_s)
            prob = clf.predict_proba(X_test_eeg_s)[:, 1]
            row['EEG_MLP_acc'] = accuracy_score(y_test, pred)
            row['EEG_MLP_f1'] = f1_score(y_test, pred, average='macro')
            row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test, pred)
            row['EEG_MLP_auroc'] = safe_roc(y_test, prob)

            clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed)
            clf.fit(X_cal_gaze_s, y_cal)
            pred = clf.predict(X_test_gaze_s)
            prob = clf.predict_proba(X_test_gaze_s)[:, 1]
            row['Gaze_MLP_acc'] = accuracy_score(y_test, pred)
            row['Gaze_MLP_f1'] = f1_score(y_test, pred, average='macro')
            row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, pred)
            row['Gaze_MLP_auroc'] = safe_roc(y_test, prob)

            X_concat_cal = np.hstack([X_cal_eeg_s, X_cal_gaze_s])
            X_concat_test = np.hstack([X_test_eeg_s, X_test_gaze_s])
            clf = RidgeClassifier(alpha=0.1)
            clf.fit(X_concat_cal, y_cal)
            pred = clf.predict(X_concat_test)
            prob = 1 / (1 + np.exp(-clf.decision_function(X_concat_test)))
            row['Concat_acc'] = accuracy_score(y_test, pred)
            row['Concat_f1'] = f1_score(y_test, pred, average='macro')
            row['Concat_bacc'] = balanced_accuracy_score(y_test, pred)
            row['Concat_auroc'] = safe_roc(y_test, prob)

            row['StaticAvg_acc'] = (row['EEG_SVM_acc'] + row['Gaze_SVM_acc']) / 2
            row['StaticAvg_f1'] = (row['EEG_SVM_f1'] + row['Gaze_SVM_f1']) / 2
            row['StaticAvg_bacc'] = (row['EEG_SVM_bacc'] + row['Gaze_SVM_bacc']) / 2
            row['StaticAvg_auroc'] = (row['EEG_SVM_auroc'] + row['Gaze_SVM_auroc']) / 2

            pcet = PCETModel()
            pcet.fit(X_cal_eeg, y_cal)
            pred = pcet.predict(X_test_eeg)
            prob = 1 / (1 + np.exp(-pcet.predict_proba_raw(X_test_eeg)))
            row['PCET_acc'] = accuracy_score(y_test, pred)
            row['PCET_f1'] = f1_score(y_test, pred, average='macro')
            row['PCET_bacc'] = balanced_accuracy_score(y_test, pred)
            row['PCET_auroc'] = safe_roc(y_test, prob)

            geta = GETAModel()
            geta.fit(X_cal_eeg, y_cal, X_cal_gaze)
            pred = geta.predict(X_test_eeg, X_test_gaze)
            prob = 1 / (1 + np.exp(-geta.predict_proba_raw(X_test_eeg, X_test_gaze)))
            row['GETA_acc'] = accuracy_score(y_test, pred)
            row['GETA_f1'] = f1_score(y_test, pred, average='macro')
            row['GETA_bacc'] = balanced_accuracy_score(y_test, pred)
            row['GETA_auroc'] = safe_roc(y_test, prob)

            cagf = CAGFFusion()
            cagf.fit(X_cal_eeg, y_cal, X_cal_gaze)
            pred, prob = cagf.predict(X_test_eeg, X_test_gaze)
            row['PCET_GETA_CAGF_acc'] = accuracy_score(y_test, pred)
            row['PCET_GETA_CAGF_f1'] = f1_score(y_test, pred, average='macro')
            row['PCET_GETA_CAGF_bacc'] = balanced_accuracy_score(y_test, pred)
            row['PCET_GETA_CAGF_auroc'] = safe_roc(y_test, prob[:, 1])

            results.append(row)
        print("OK", flush=True)

    return results

def main():
    print("="*70)
    print("Few-Shot AdaGTCN-Proxy Comparison (Simplified)")
    print("="*70)
    print("\nNote: This version runs without neural network models.")
    print("For full GCN models, run fewshot_adagtcn_proxy_comparison.py\n")

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_few_shot_experiment(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "fewshot_adagtcn_proxy_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP',
               'Concat', 'StaticAvg', 'PCET', 'GETA', 'PCET_GETA_CAGF']
    metrics = ['acc', 'f1', 'bacc', 'auroc']

    summary_data = []
    for method in methods:
        for k in k_values:
            subset = df[df['k'] == k]
            if len(subset) == 0:
                continue
            row = {'Method': method, 'k': k}
            for metric in metrics:
                col = f'{method}_{metric}'
                if col in subset.columns:
                    mean_val = subset[col].mean() * 100
                    std_val = subset[col].std() * 100
                    row[metric.upper()] = f"{mean_val:.1f}±{std_val:.1f}"
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULTS_DIR, "fewshot_adagtcn_proxy_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))

    report = f"""# Few-Shot AdaGTCN-Proxy Comparison Report (Simplified)

## Important Note
**This is an AdaGTCN-inspired proxy under our few-shot protocol, not a full reproduction of AdaGTCN.**

## Experiment Protocol
- LOSO target subject
- For each target subject: calibration pool内每类采样k-shot
- k = {k_values}
- Test on remaining target-subject samples
- seeds = {seeds}

## Methods (Simplified Baselines)
1. **EEG_SVM**: Ridge Classifier on EEG features
2. **Gaze_SVM**: Ridge Classifier on Gaze features
3. **EEG_MLP**: MLP Classifier on EEG features
4. **Gaze_MLP**: MLP Classifier on Gaze features
5. **Concat**: Ridge on concatenated EEG+Gaze
6. **StaticAvg**: Average of EEG_SVM and Gaze_SVM
7. **PCET**: PCA reconstruction error features
8. **GETA**: Gaze-guided EEG attention
9. **PCET+GETA+CAGF**: Full proposed model

## Key Questions Answered

### 1. AdaGTCN-proxy在3/5/10/20/50-shot下是多少？
(GCN models not included in this simplified version)

### 2. EEG-GCN-proxy是否强于EEG-MLP？
(GCN models not included in this simplified version)

### 3. EEG-GCN+Gaze-MLP-proxy是否强于EEG+Gaze concat？
(GCN models not included in this simplified version)

### 4. 我们的PCET+GETA+CAGF是否超过这些proxy baseline？
See summary table.

### 5. 如果没有超过，在哪些shot下没有超过？
Analyze PCET_GETA_CAGF vs other methods.

### 6. 这些结果是否支持论文主打few-shot personalized calibration？
If PCET_GETA_CAGF exceeds baselines at higher shots.

## Results

{summary_df.to_string(index=False)}
"""

    report_path = os.path.join(REPORTS_DIR, "fewshot_adagtcn_proxy_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")
    print("\nDONE!")

if __name__ == '__main__':
    main()