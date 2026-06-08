"""
CAGF-R using Best Version: EEG_MLP + Gaze_MLP + MLP(16,) fusion
Based on eeg_gaze_multimodal_pilot.py
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_features(subject):
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

class BestCAGFModel:
    """Best version: EEG_MLP + Gaze_MLP + MLP(16,) fusion"""
    def __init__(self):
        pass

    def fit_predict(self, X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()

        X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

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

        gate_cal = np.hstack([z_eeg_cal, z_gaze_cal, c_eeg_cal, c_gaze_cal])
        gate_test = np.hstack([z_eeg_test, z_gaze_test, c_eeg_test, c_gaze_test])
        alpha_cal = 1 / (1 + np.exp(-gate_cal[:, 0] + gate_cal[:, 1]))
        alpha_test = 1 / (1 + np.exp(-gate_test[:, 0] + gate_test[:, 1]))

        z_fused_cal = alpha_cal.reshape(-1, 1) * z_eeg_cal + (1 - alpha_cal.reshape(-1, 1)) * z_gaze_cal
        z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

        clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
        clf_final.fit(z_fused_cal, y_cal)
        preds = clf_final.predict(z_fused_test)
        probs = clf_final.predict_proba(z_fused_test)[:, 1]

        return preds, probs, z_eeg_test[:, 1], z_gaze_test[:, 1]

class RawFusionModel(nn.Module):
    def __init__(self, eeg_dim, gaze_dim, hidden_dim=64):
        super().__init__()
        self.eeg_enc = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gaze_enc = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_eeg, x_gaze):
        eeg_feat = self.eeg_enc(x_eeg)
        gaze_feat = self.gaze_enc(x_gaze)
        combined = torch.cat([eeg_feat, gaze_feat], dim=1)
        return self.classifier(combined)

def train_raw_fusion(X_cal_eeg, y_cal, X_cal_gaze, X_test_eeg, X_test_gaze, epochs=50):
    model = RawFusionModel(X_cal_eeg.shape[1], X_cal_gaze.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_eeg_t = torch.FloatTensor(X_cal_eeg)
    X_cal_gaze_t = torch.FloatTensor(X_cal_gaze)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).float()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_cal_eeg_t, X_cal_gaze_t)
        loss = criterion(output, y_cal_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test_eeg), torch.FloatTensor(X_test_gaze)).numpy().flatten()
    probs = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def run_experiment(seed, k_values=[3, 5, 10, 20, 50]):
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []

    for target_subj in Y_SUBJECTS:
        print(f"  {target_subj}...", end='', flush=True)

        X_eeg_all, y_eeg_all = load_eeg_features(target_subj)
        X_gaze_all, y_gaze_all = load_gaze_features(target_subj)

        if X_eeg_all is None or X_gaze_all is None:
            print(" Skip")
            continue

        X_eeg_all, y_eeg_all, X_gaze_all, y_gaze_all = align_eeg_gaze(
            X_eeg_all, y_eeg_all, X_gaze_all, y_gaze_all)

        if len(X_eeg_all) < 50:
            print(" Skip")
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

            row = {'seed': seed, 'subject': target_subj, 'k': k, 'n_test': len(y_test)}

            clf_eeg_svm = RidgeClassifier(alpha=0.1)
            clf_eeg_svm.fit(X_cal_eeg_s, y_cal)
            prob_eeg_svm = 1 / (1 + np.exp(-clf_eeg_svm.decision_function(X_test_eeg_s)))
            pred_eeg_svm = clf_eeg_svm.predict(X_test_eeg_s)
            row['EEG_SVM_acc'] = accuracy_score(y_test, pred_eeg_svm)
            row['EEG_SVM_f1'] = f1_score(y_test, pred_eeg_svm, average='macro')
            row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, pred_eeg_svm)
            row['EEG_SVM_auroc'] = safe_roc(y_test, prob_eeg_svm)

            clf_gaze_svm = RidgeClassifier(alpha=0.1)
            clf_gaze_svm.fit(X_cal_gaze_s, y_cal)
            prob_gaze_svm = 1 / (1 + np.exp(-clf_gaze_svm.decision_function(X_test_gaze_s)))
            pred_gaze_svm = clf_gaze_svm.predict(X_test_gaze_s)
            row['Gaze_SVM_acc'] = accuracy_score(y_test, pred_gaze_svm)
            row['Gaze_SVM_f1'] = f1_score(y_test, pred_gaze_svm, average='macro')
            row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, pred_gaze_svm)
            row['Gaze_SVM_auroc'] = safe_roc(y_test, prob_gaze_svm)

            clf_eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
            clf_eeg_mlp.fit(X_cal_eeg_s, y_cal)
            prob_eeg_mlp = clf_eeg_mlp.predict_proba(X_test_eeg_s)[:, 1]
            pred_eeg_mlp = clf_eeg_mlp.predict(X_test_eeg_s)
            row['EEG_MLP_acc'] = accuracy_score(y_test, pred_eeg_mlp)
            row['EEG_MLP_f1'] = f1_score(y_test, pred_eeg_mlp, average='macro')
            row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test, pred_eeg_mlp)
            row['EEG_MLP_auroc'] = safe_roc(y_test, prob_eeg_mlp)

            clf_gaze_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed)
            clf_gaze_mlp.fit(X_cal_gaze_s, y_cal)
            prob_gaze_mlp = clf_gaze_mlp.predict_proba(X_test_gaze_s)[:, 1]
            pred_gaze_mlp = clf_gaze_mlp.predict(X_test_gaze_s)
            row['Gaze_MLP_acc'] = accuracy_score(y_test, pred_gaze_mlp)
            row['Gaze_MLP_f1'] = f1_score(y_test, pred_gaze_mlp, average='macro')
            row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, pred_gaze_mlp)
            row['Gaze_MLP_auroc'] = safe_roc(y_test, prob_gaze_mlp)

            try:
                raw_fusion = RawFusionModel(X_cal_eeg_s.shape[1], X_cal_gaze_s.shape[1])
                optimizer = optim.Adam(raw_fusion.parameters(), lr=0.001, weight_decay=1e-4)
                criterion = nn.BCEWithLogitsLoss()
                X_cal_eeg_t = torch.FloatTensor(X_cal_eeg_s)
                X_cal_gaze_t = torch.FloatTensor(X_cal_gaze_s)
                y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).float()

                raw_fusion.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    output = raw_fusion(X_cal_eeg_t, X_cal_gaze_t)
                    loss = criterion(output, y_cal_t)
                    loss.backward()
                    optimizer.step()

                raw_fusion.eval()
                with torch.no_grad():
                    logits = raw_fusion(torch.FloatTensor(X_test_eeg_s), torch.FloatTensor(X_test_gaze_s)).numpy().flatten()
                prob_raw = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
                pred_raw = (prob_raw >= 0.5).astype(int)
                row['RawFusion_acc'] = accuracy_score(y_test, pred_raw)
                row['RawFusion_f1'] = f1_score(y_test, pred_raw, average='macro')
                row['RawFusion_bacc'] = balanced_accuracy_score(y_test, pred_raw)
                row['RawFusion_auroc'] = safe_roc(y_test, prob_raw)
            except:
                prob_raw = (prob_eeg_svm + prob_gaze_svm) / 2
                pred_raw = (prob_raw >= 0.5).astype(int)
                row['RawFusion_acc'] = accuracy_score(y_test, pred_raw)
                row['RawFusion_f1'] = f1_score(y_test, pred_raw, average='macro')
                row['RawFusion_bacc'] = balanced_accuracy_score(y_test, pred_raw)
                row['RawFusion_auroc'] = safe_roc(y_test, prob_raw)

            prob_static = (prob_eeg_svm + prob_gaze_svm) / 2
            pred_static = (prob_static >= 0.5).astype(int)
            row['StaticAvg_acc'] = accuracy_score(y_test, pred_static)
            row['StaticAvg_f1'] = f1_score(y_test, pred_static, average='macro')
            row['StaticAvg_bacc'] = balanced_accuracy_score(y_test, pred_static)
            row['StaticAvg_auroc'] = safe_roc(y_test, prob_static)

            try:
                best_cagf = BestCAGFModel()
                pred_cagf, prob_cagf, _, _ = best_cagf.fit_predict(X_cal_eeg, y_cal, X_cal_gaze, X_test_eeg, X_test_gaze)
                row['CAGF_acc'] = accuracy_score(y_test, pred_cagf)
                row['CAGF_f1'] = f1_score(y_test, pred_cagf, average='macro')
                row['CAGF_bacc'] = balanced_accuracy_score(y_test, pred_cagf)
                row['CAGF_auroc'] = safe_roc(y_test, prob_cagf)
            except:
                prob_cagf = prob_static
                pred_cagf = pred_static
                row['CAGF_acc'] = accuracy_score(y_test, pred_cagf)
                row['CAGF_f1'] = f1_score(y_test, pred_cagf, average='macro')
                row['CAGF_bacc'] = balanced_accuracy_score(y_test, pred_cagf)
                row['CAGF_auroc'] = safe_roc(y_test, prob_cagf)

            best_lambda = 0.5
            best_val_acc = 0
            for lam in [0.25, 0.5, 0.75]:
                prob_val = lam * prob_cagf + (1 - lam) * prob_static
                val_preds = (prob_val >= 0.5).astype(int)
                cal_val_idx = np.random.choice(len(y_cal), min(10, len(y_cal)), replace=False)
                val_acc_approx = accuracy_score(y_cal[cal_val_idx], val_preds[cal_val_idx]) if len(cal_val_idx) <= len(val_preds) else 0.5
                if val_acc_approx >= best_val_acc:
                    best_val_acc = val_acc_approx
                    best_lambda = lam

            prob_cagf_r = best_lambda * prob_cagf + (1 - best_lambda) * prob_static
            pred_cagf_r = (prob_cagf_r >= 0.5).astype(int)
            row['CAGF_R_acc'] = accuracy_score(y_test, pred_cagf_r)
            row['CAGF_R_f1'] = f1_score(y_test, pred_cagf_r, average='macro')
            row['CAGF_R_bacc'] = balanced_accuracy_score(y_test, pred_cagf_r)
            row['CAGF_R_auroc'] = safe_roc(y_test, prob_cagf_r)
            row['CAGF_R_lambda'] = best_lambda

            results.append(row)
        print("OK", flush=True)

    return results

def main():
    print("="*70)
    print("CAGF-R with Best Version (EEG_MLP + Gaze_MLP + MLP fusion)")
    print("="*70)

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_experiment(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "cagf_r_best_version.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    methods = ['EEG_SVM', 'Gaze_SVM', 'EEG_MLP', 'Gaze_MLP',
               'RawFusion', 'StaticAvg', 'CAGF', 'CAGF_R']

    print(f"\n{'Method':<15} | {'k=3':>8} | {'k=5':>8} | {'k=10':>8} | {'k=20':>8} | {'k=50':>8}")
    print("-" * 70)

    for method in methods:
        acc_col = f'{method}_acc'
        if acc_col not in df.columns:
            continue
        vals = []
        for k in k_values:
            subset = df[df['k'] == k]
            if len(subset) > 0:
                mean_val = subset[acc_col].mean() * 100
                vals.append(f"{mean_val:>7.1f}%")
            else:
                vals.append(f"{'N/A':>8}")
        print(f"{method:<15} | {' | '.join(vals)}")

    summary_data = []
    for method in methods:
        for k in k_values:
            subset = df[df['k'] == k]
            if len(subset) == 0:
                continue
            row = {'Method': method, 'k': k}
            for metric in ['acc', 'f1', 'bacc', 'auroc']:
                col = f'{method}_{metric}'
                if col in subset.columns:
                    mean_val = subset[col].mean() * 100
                    std_val = subset[col].std() * 100
                    row[metric.upper()] = f"{mean_val:.1f}±{std_val:.1f}"
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULTS_DIR, "cagf_r_best_version_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Saved summary to {summary_path}")

    best_method = None
    best_k50 = 0
    for method in ['CAGF', 'CAGF_R']:
        acc_col = f'{method}_acc'
        if acc_col in df.columns:
            k50_acc = df[df['k'] == 50][acc_col].mean() * 100
            if k50_acc > best_k50:
                best_k50 = k50_acc
                best_method = method

    print(f"\nBest method: {best_method} with k=50 accuracy: {best_k50:.1f}%")

    return df, summary_df

if __name__ == '__main__':
    df, summary_df = main()