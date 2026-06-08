"""
New CAGF Designs for Better EEG-Gaze Fusion

Problem with current CAGF:
- Output-level fusion only
- Collapses to simple average when predictions are similar
- Doesn't leverage feature-level interaction

New designs:
1. CAGF_v2: Confidence-Weighted Disagreement Fusion
2. CAGF_v3: Feature-Level Cross Attention Fusion
3. CAGF_v4: Learned Routing with Calibration
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

        entropy = -np.sum(z_gaze_prob * np.log(z_gaze_prob + 1e-3), axis=1).reshape(-1, 1)
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

class CAGFv2_ConfidenceDisagreement:
    """
    New CAGF with Confidence-Weighted Disagreement Fusion

    Key insight: When EEG and Gaze predictions disagree, we should
    dynamically choose which to trust based on:
    1. Per-modality confidence
    2. Prediction disagreement magnitude
    """
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

        p_pcet = 1 / (1 + np.exp(-z_pcet))
        p_geta = 1 / (1 + np.exp(-z_geta))

        confidence_pcet = 2 * np.abs(p_pcet - 0.5)
        confidence_pgeta = 2 * np.abs(p_geta - 0.5)

        disagreement = np.abs(p_pcet - p_geta)

        alpha = 0.5 * np.ones_like(p_pcet)

        strong_disagree_mask = disagreement > 0.3
        high_conf_eeg_mask = confidence_pcet > confidence_pgeta

        alpha[strong_disagree_mask & high_conf_eeg_mask] = 0.8
        alpha[strong_disagree_mask & ~high_conf_eeg_mask] = 0.2

        soft_disagree_mask = (disagreement > 0.1) & (disagreement <= 0.3)
        conf_diff = np.abs(confidence_pcet - confidence_pgeta)
        alpha[soft_disagree_mask] = 0.5 + 0.3 * (confidence_pcet[soft_disagree_mask] - confidence_pgeta[soft_disagree_mask]) / (conf_diff[soft_disagree_mask] + 1e-6)

        alpha = np.clip(alpha, 0.1, 0.9)

        p_fused = alpha * p_pcet + (1 - alpha) * p_geta
        return (p_fused >= 0.5).astype(int), np.column_stack([1-p_fused, p_fused])

class CAGFv3_FeatureConcatMLP:
    """
    CAGF v3: Feature-level fusion with MLP

    Instead of output-level fusion, concatenate EEG and Gaze features
    and train an MLP to learn the optimal fusion
    """
    def __init__(self):
        self.scaler_eeg = StandardScaler()
        self.scaler_gaze = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    def fit(self, X_eeg_train, y_train, X_gaze_train):
        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg_train)
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze_train)
        X_combined = np.hstack([X_eeg_s, X_gaze_s])
        self.clf.fit(X_combined, y_train)
        return self

    def predict(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)
        X_combined = np.hstack([X_eeg_s, X_gaze_s])
        probs = self.clf.predict_proba(X_combined)
        return (probs[:, 1] >= 0.5).astype(int), probs

    def predict_proba(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)
        X_combined = np.hstack([X_eeg_s, X_gaze_s])
        probs = self.clf.predict_proba(X_combined)
        return (probs[:, 1] >= 0.5).astype(int), probs

class CAGFv4_EnsembleWithRouting:
    """
    CAGF v4: Ensemble with learned routing

    Train separate EEG and Gaze models, then use a routing function
    that decides which model's prediction to trust based on
    the input features
    """
    def __init__(self):
        self.pcet = PCETModel()
        self.geta = GETAModel()
        self.scaler_eeg = StandardScaler()
        self.scaler_gaze = StandardScaler()
        self.router = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)

    def fit(self, X_eeg_train, y_train, X_gaze_train):
        self.pcet.fit(X_eeg_train, y_train)
        self.geta.fit(X_eeg_train, y_train, X_gaze_train)

        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg_train)
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze_train)

        z_pcet = self.pcet.predict_proba_raw(X_eeg_train)
        z_geta = self.geta.predict_proba_raw(X_eeg_train, X_gaze_train)

        p_pcet = 1 / (1 + np.exp(-z_pcet))
        p_geta = 1 / (1 + np.exp(-z_geta))

        disagreement = np.abs(p_pcet - p_geta).reshape(-1, 1)
        conf_diff = (np.abs(p_pcet - 0.5) - np.abs(p_geta - 0.5)).reshape(-1, 1)

        X_route = np.hstack([
            X_eeg_s[:, :20],
            X_gaze_s,
            disagreement,
            conf_diff
        ])

        y_route = (p_pcet > p_geta).astype(int)
        self.router.fit(X_route, y_route)

        return self

    def predict(self, X_eeg_test, X_gaze_test):
        X_eeg_s = self.scaler_eeg.transform(X_eeg_test)
        X_gaze_s = self.scaler_gaze.transform(X_gaze_test)

        z_pcet = self.pcet.predict_proba_raw(X_eeg_test)
        z_geta = self.geta.predict_proba_raw(X_eeg_test, X_gaze_test)

        p_pcet = 1 / (1 + np.exp(-z_pcet))
        p_geta = 1 / (1 + np.exp(-z_geta))

        disagreement = np.abs(p_pcet - p_geta).reshape(-1, 1)
        conf_diff = (np.abs(p_pcet - 0.5) - np.abs(p_geta - 0.5)).reshape(-1, 1)

        X_route = np.hstack([
            X_eeg_s[:, :20],
            X_gaze_s,
            disagreement,
            conf_diff
        ])

        route = self.router.predict(X_route)

        pred_pcet = (p_pcet >= 0.5).astype(int)
        pred_geta = (p_geta >= 0.5).astype(int)

        preds = np.where(route == 1, pred_pcet, pred_geta)

        p_fused = np.where(route.reshape(-1, 1) == 1,
                          np.column_stack([1-p_pcet, p_pcet]),
                          np.column_stack([1-p_geta, p_geta]))

        return preds, p_fused

def run_comparison(seed, k_values=[3, 5, 10, 20, 50]):
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

            cagf_v1 = CAGFFusion()
            cagf_v1.fit(X_cal_eeg, y_cal, X_cal_gaze)
            pred_v1, prob_v1 = cagf_v1.predict(X_test_eeg, X_test_gaze)
            row['CAGF_v1_acc'] = accuracy_score(y_test, pred_v1)
            row['CAGF_v1_f1'] = f1_score(y_test, pred_v1, average='macro')
            row['CAGF_v1_bacc'] = balanced_accuracy_score(y_test, pred_v1)
            row['CAGF_v1_auroc'] = safe_roc(y_test, prob_v1[:, 1])

            cagf_v2 = CAGFv2_ConfidenceDisagreement()
            cagf_v2.fit(X_cal_eeg, y_cal, X_cal_gaze)
            pred_v2, prob_v2 = cagf_v2.predict(X_test_eeg, X_test_gaze)
            row['CAGF_v2_acc'] = accuracy_score(y_test, pred_v2)
            row['CAGF_v2_f1'] = f1_score(y_test, pred_v2, average='macro')
            row['CAGF_v2_bacc'] = balanced_accuracy_score(y_test, pred_v2)
            row['CAGF_v2_auroc'] = safe_roc(y_test, prob_v2[:, 1])

            cagf_v3 = CAGFv3_FeatureConcatMLP()
            cagf_v3.fit(X_cal_eeg, y_cal, X_cal_gaze)
            pred_v3, prob_v3 = cagf_v3.predict(X_test_eeg, X_test_gaze)
            row['CAGF_v3_acc'] = accuracy_score(y_test, pred_v3)
            row['CAGF_v3_f1'] = f1_score(y_test, pred_v3, average='macro')
            row['CAGF_v3_bacc'] = balanced_accuracy_score(y_test, pred_v3)
            row['CAGF_v3_auroc'] = safe_roc(y_test, prob_v3[:, 1])

            cagf_v4 = CAGFv4_EnsembleWithRouting()
            try:
                cagf_v4.fit(X_cal_eeg, y_cal, X_cal_gaze)
                pred_v4, prob_v4 = cagf_v4.predict(X_test_eeg, X_test_gaze)
                row['CAGF_v4_acc'] = accuracy_score(y_test, pred_v4)
                row['CAGF_v4_f1'] = f1_score(y_test, pred_v4, average='macro')
                row['CAGF_v4_bacc'] = balanced_accuracy_score(y_test, pred_v4)
                row['CAGF_v4_auroc'] = safe_roc(y_test, prob_v4[:, 1])
            except:
                row['CAGF_v4_acc'] = 0.5
                row['CAGF_v4_f1'] = 0.5
                row['CAGF_v4_bacc'] = 0.5
                row['CAGF_v4_auroc'] = 0.5

            results.append(row)
        print("OK", flush=True)

    return results

def main():
    print("="*70)
    print("CAGF Comparison: v1 (original) vs v2, v3, v4 (new designs)")
    print("="*70)

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_comparison(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "cagf_comparison.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("CAGF COMPARISON RESULTS")
    print("="*70)

    methods = ['CAGF_v1', 'CAGF_v2', 'CAGF_v3', 'CAGF_v4']
    metrics = ['acc', 'f1', 'bacc', 'auroc']

    for k in k_values:
        print(f"\n### k={k}")
        subset = df[df['k'] == k]
        for method in methods:
            acc_col = f'{method}_acc'
            if acc_col in subset.columns:
                mean_val = subset[acc_col].mean() * 100
                std_val = subset[acc_col].std() * 100
                print(f"  {method}: {mean_val:.1f}±{std_val:.1f}%")

    print(f"\nSaved to {output_path}")

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
    summary_path = os.path.join(RESULTS_DIR, "cagf_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    for method in methods[1:]:
        v1_col = f'CAGF_v1_acc'
        vx_col = f'{method}_acc'
        if v1_col in df.columns and vx_col in df.columns:
            diff = (df[vx_col].mean() - df[v1_col].mean()) * 100
            print(f"{method} vs v1: {diff:+.1f}% improvement")

    print("\nDONE!")

if __name__ == '__main__':
    main()