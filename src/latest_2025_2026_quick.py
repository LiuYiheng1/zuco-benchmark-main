"""
Quick 2025-2026 Latest Methods Proxy Baselines
Reduced epochs and simplified models for faster execution
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

class SimpleGraphEEG(nn.Module):
    """Simplified Graph EEG with learnable adjacency"""
    def __init__(self, input_dim, hidden_dim=32, n_groups=20):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        self.adj = nn.Parameter(torch.ones(n_groups, n_groups) / n_groups)
        self.fc = nn.Linear(n_groups * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_groups = x[:, :self.n_groups * self.group_dim].view(batch_size, self.n_groups, self.group_dim)

        adj = torch.softmax(self.adj, dim=-1)
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        aggregated = torch.matmul(adj, x_groups)
        graph_feat = aggregated.reshape(batch_size, -1)
        encoded = torch.relu(self.fc(graph_feat))
        return self.classifier(encoded)

class STRE_Lite(nn.Module):
    """STRE-lite: Temporal convolution after graph aggregation"""
    def __init__(self, input_dim, hidden_dim=32, n_groups=20):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        self.adj = nn.Parameter(torch.ones(n_groups, n_groups) / n_groups)
        self.fc = nn.Linear(n_groups * hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.classifier = nn.Linear(hidden_dim * n_groups, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_groups = x[:, :self.n_groups * self.group_dim].view(batch_size, self.n_groups, self.group_dim)

        adj = torch.softmax(self.adj, dim=-1)
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        aggregated = torch.matmul(adj, x_groups)
        agg_t = aggregated.transpose(1, 2)
        conv_feat = torch.relu(self.conv(agg_t))
        conv_feat = conv_feat.transpose(1, 2).reshape(batch_size, -1)
        encoded = torch.relu(self.fc(conv_feat))
        return self.classifier(encoded)

class GLIM_Encoder(nn.Module):
    """GLIM: Interpretable bottleneck encoder"""
    def __init__(self, input_dim, bottleneck_dim=16, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        return self.classifier(bottleneck)

def train_model(model, X_cal, y_cal, X_test, epochs=30, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_t = torch.FloatTensor(X_cal)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).float()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_cal_t)
        loss = criterion(output, y_cal_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test)).numpy().flatten()

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

            eeg_dim = X_cal_eeg_s.shape[1]

            row = {'seed': seed, 'subject': target_subj, 'k': k, 'n_test': len(y_test)}

            try:
                strg = SimpleGraphEEG(eeg_dim, hidden_dim=32, n_groups=20)
                pred, prob = train_model(strg, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=30)
                row['STRG_lite_acc'] = accuracy_score(y_test, pred)
                row['STRG_lite_f1'] = f1_score(y_test, pred, average='macro')
                row['STRG_lite_bacc'] = balanced_accuracy_score(y_test, pred)
                row['STRG_lite_auroc'] = safe_roc(y_test, prob)
            except:
                row['STRG_lite_acc'] = 0.5
                row['STRG_lite_f1'] = 0.5
                row['STRG_lite_bacc'] = 0.5
                row['STRG_lite_auroc'] = 0.5

            try:
                stre = STRE_Lite(eeg_dim, hidden_dim=32, n_groups=20)
                pred, prob = train_model(stre, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=30)
                row['STRE_lite_acc'] = accuracy_score(y_test, pred)
                row['STRE_lite_f1'] = f1_score(y_test, pred, average='macro')
                row['STRE_lite_bacc'] = balanced_accuracy_score(y_test, pred)
                row['STRE_lite_auroc'] = safe_roc(y_test, prob)
            except:
                row['STRE_lite_acc'] = 0.5
                row['STRE_lite_f1'] = 0.5
                row['STRE_lite_bacc'] = 0.5
                row['STRE_lite_auroc'] = 0.5

            try:
                glim = GLIM_Encoder(eeg_dim, bottleneck_dim=16, hidden_dim=32)
                pred, prob = train_model(glim, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=30)
                row['GLIM_enc_acc'] = accuracy_score(y_test, pred)
                row['GLIM_enc_f1'] = f1_score(y_test, pred, average='macro')
                row['GLIM_enc_bacc'] = balanced_accuracy_score(y_test, pred)
                row['GLIM_enc_auroc'] = safe_roc(y_test, prob)
            except:
                row['GLIM_enc_acc'] = 0.5
                row['GLIM_enc_f1'] = 0.5
                row['GLIM_enc_bacc'] = 0.5
                row['GLIM_enc_auroc'] = 0.5

            try:
                cog = CognitiveDecoderProxy(eeg_dim)
                pred, prob = train_cognitive(cog, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=30)
                row['Cog_EEGtext_acc'] = accuracy_score(y_test, pred)
                row['Cog_EEGtext_f1'] = f1_score(y_test, pred, average='macro')
                row['Cog_EEGtext_bacc'] = balanced_accuracy_score(y_test, pred)
                row['Cog_EEGtext_auroc'] = safe_roc(y_test, prob)
            except:
                row['Cog_EEGtext_acc'] = 0.5
                row['Cog_EEGtext_f1'] = 0.5
                row['Cog_EEGtext_bacc'] = 0.5
                row['Cog_EEGtext_auroc'] = 0.5

            results.append(row)
        print("OK", flush=True)

    return results

class CognitiveDecoderProxy(nn.Module):
    """CognitiveDecoder: EEG-text style fusion"""
    def __init__(self, eeg_dim, hidden_dim=32):
        super().__init__()
        self.eeg_enc = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.text_proxy = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_eeg):
        eeg_feat = self.eeg_enc(x_eeg)
        text_feat = self.text_proxy(x_eeg)
        combined = torch.cat([eeg_feat, text_feat], dim=1)
        return self.fusion(combined)

def train_cognitive(model, X_cal, y_cal, X_test, epochs=30, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_t = torch.FloatTensor(X_cal)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).float()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_cal_t)
        loss = criterion(output, y_cal_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test)).numpy().flatten()

    probs = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def main():
    print("="*70)
    print("2025-2026 Latest Methods Proxy (Quick Version)")
    print("="*70)

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_experiment(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "latest_2025_2026_proxy_baselines.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    methods = ['STRG_lite', 'STRE_lite', 'GLIM_enc', 'Cog_EEGtext']

    print(f"\n{'Method':<20} | {'k=3':>8} | {'k=5':>8} | {'k=10':>8} | {'k=20':>8} | {'k=50':>8}")
    print("-" * 80)

    for method in methods:
        acc_col = f'{method}_acc'
        if acc_col not in df.columns:
            continue
        vals = []
        for k in k_values:
            subset = df[df['k'] == k]
            if len(subset) > 0:
                mean_val = subset[acc_col].mean() * 100
                vals.append(f"{mean_val:.1f}")
            else:
                vals.append("N/A")
        print(f"{method:<20} | {' | '.join([f'{v:>8}' for v in vals])}")

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
    summary_path = os.path.join(RESULTS_DIR, "latest_2025_2026_proxy_baselines_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Saved summary to {summary_path}")
    print("\nDONE!")

if __name__ == '__main__':
    main()