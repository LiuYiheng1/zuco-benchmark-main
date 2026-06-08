"""
Few-Shot AdaGTCN-Proxy Comparison - Full Version with GCN
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

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        return self.linear(torch.matmul(adj, x))

class EEGGCNProxy(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_groups=20):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        self.fc_self = nn.Linear(self.group_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(self.group_dim, hidden_dim)
        self.fc_graph = nn.Linear(n_groups * hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_groups = x[:, :self.n_groups * self.group_dim].view(batch_size, self.n_groups, self.group_dim)

        adj = torch.softmax(torch.matmul(x_groups, x_groups.transpose(1, 2)), dim=-1)

        self_feat = self.fc_self(x_groups)
        neigh_feat = self.fc_neighbor(x_groups)
        aggregated = self_feat + torch.matmul(adj, neigh_feat)

        graph_feat = aggregated.reshape(batch_size, -1)
        graph_encoded = self.fc_graph(graph_feat)

        return self.classifier(graph_encoded)

class GazeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.net(x)
        return self.classifier(feat)

class EEGGCNGazeMLPProxy(nn.Module):
    def __init__(self, eeg_dim, gaze_dim, hidden_dim=64):
        super().__init__()
        self.eeg_gcn = EEGGCNProxy(eeg_dim, hidden_dim)
        self.gaze_mlp = GazeMLP(gaze_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_eeg, x_gaze):
        eeg_out = self.eeg_gcn(x_eeg)
        gaze_out = self.gaze_mlp(x_gaze)
        combined = torch.cat([eeg_out, gaze_out], dim=1)
        return self.fusion(combined)

class AdaGTCNLite(nn.Module):
    def __init__(self, eeg_dim, gaze_dim, hidden_dim=64):
        super().__init__()
        self.eeg_adj = nn.Linear(eeg_dim, eeg_dim)
        self.eeg_gcn1 = EEGGCNProxy(eeg_dim, hidden_dim)
        self.eeg_conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.gaze_mlp = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_eeg, x_gaze):
        adj_weight = torch.sigmoid(self.eeg_adj(x_eeg))
        eeg_feat = self.eeg_gcn1(x_eeg * adj_weight)
        eeg_feat = eeg_feat.unsqueeze(2)
        eeg_feat = self.eeg_conv1d(eeg_feat).squeeze(2)

        gaze_feat = self.gaze_mlp(x_gaze)

        combined = torch.cat([eeg_feat, gaze_feat], dim=1)
        return self.fusion(combined)

def train_neural_model(model, X_cal_eeg, y_cal, X_test_eeg, X_cal_gaze=None, X_test_gaze=None,
                       epochs=30, lr=0.001, batch_size=32, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_eeg_t = torch.FloatTensor(X_cal_eeg).to(device)
    y_cal_t = torch.FloatTensor(y_cal).to(device).unsqueeze(1)
    X_test_eeg_t = torch.FloatTensor(X_test_eeg).to(device)

    if X_cal_gaze is not None:
        X_cal_gaze_t = torch.FloatTensor(X_cal_gaze).to(device)
        X_test_gaze_t = torch.FloatTensor(X_test_gaze).to(device)

    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(len(y_cal_t))
        for i in range(0, len(y_cal_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_eeg = X_cal_eeg_t[batch_idx]
            batch_y = y_cal_t[batch_idx]

            optimizer.zero_grad()
            if X_cal_gaze is not None:
                batch_gaze = X_cal_gaze_t[batch_idx]
                output = model(batch_eeg, batch_gaze)
            else:
                output = model(batch_eeg)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        if X_cal_gaze is not None:
            test_logits = model(X_test_eeg_t, X_test_gaze_t).cpu().numpy().flatten()
        else:
            test_logits = model(X_test_eeg_t).cpu().numpy().flatten()

    test_probs = 1 / (1 + np.exp(-test_logits))
    test_preds = (test_probs >= 0.5).astype(int)
    return test_preds, test_probs

def safe_roc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5

def run_gcn_experiment(seed, k_values=[3, 5, 10, 20, 50]):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'

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

            eeg_dim = X_cal_eeg_s.shape[1]
            gaze_dim = X_cal_gaze_s.shape[1]

            row = {
                'seed': seed, 'subject': target_subj, 'k': k,
                'n_test': len(y_test), 'n_cal': len(y_cal),
            }

            try:
                eeg_gcn = EEGGCNProxy(eeg_dim, hidden_dim=64, n_groups=20).to(device)
                pred, prob = train_neural_model(
                    eeg_gcn, X_cal_eeg_s, y_cal, X_test_eeg_s,
                    epochs=30, device=device)
                row['EEG_GCN_acc'] = accuracy_score(y_test, pred)
                row['EEG_GCN_f1'] = f1_score(y_test, pred, average='macro')
                row['EEG_GCN_bacc'] = balanced_accuracy_score(y_test, pred)
                row['EEG_GCN_auroc'] = safe_roc(y_test, prob)
            except Exception as e:
                row['EEG_GCN_acc'] = 0.5
                row['EEG_GCN_f1'] = 0.5
                row['EEG_GCN_bacc'] = 0.5
                row['EEG_GCN_auroc'] = 0.5

            try:
                eeg_gcn_gaze = EEGGCNGazeMLPProxy(eeg_dim, gaze_dim, hidden_dim=64).to(device)
                pred, prob = train_neural_model(
                    eeg_gcn_gaze, X_cal_eeg_s, y_cal, X_test_eeg_s,
                    X_cal_gaze_s, X_test_gaze_s, epochs=30, device=device)
                row['EEG_GCN_Gaze_MLP_acc'] = accuracy_score(y_test, pred)
                row['EEG_GCN_Gaze_MLP_f1'] = f1_score(y_test, pred, average='macro')
                row['EEG_GCN_Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, pred)
                row['EEG_GCN_Gaze_MLP_auroc'] = safe_roc(y_test, prob)
            except Exception as e:
                row['EEG_GCN_Gaze_MLP_acc'] = 0.5
                row['EEG_GCN_Gaze_MLP_f1'] = 0.5
                row['EEG_GCN_Gaze_MLP_bacc'] = 0.5
                row['EEG_GCN_Gaze_MLP_auroc'] = 0.5

            try:
                adagtcn_lite = AdaGTCNLite(eeg_dim, gaze_dim, hidden_dim=64).to(device)
                pred, prob = train_neural_model(
                    adagtcn_lite, X_cal_eeg_s, y_cal, X_test_eeg_s,
                    X_cal_gaze_s, X_test_gaze_s, epochs=30, device=device)
                row['AdaGTCN_lite_acc'] = accuracy_score(y_test, pred)
                row['AdaGTCN_lite_f1'] = f1_score(y_test, pred, average='macro')
                row['AdaGTCN_lite_bacc'] = balanced_accuracy_score(y_test, pred)
                row['AdaGTCN_lite_auroc'] = safe_roc(y_test, prob)
            except Exception as e:
                row['AdaGTCN_lite_acc'] = 0.5
                row['AdaGTCN_lite_f1'] = 0.5
                row['AdaGTCN_lite_bacc'] = 0.5
                row['AdaGTCN_lite_auroc'] = 0.5

            results.append(row)
        print("OK", flush=True)

    return results

def main():
    print("="*70)
    print("Few-Shot AdaGTCN-Proxy Comparison (GCN Models)")
    print("="*70)

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_gcn_experiment(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "fewshot_gcn_proxy_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    methods = ['EEG_GCN', 'EEG_GCN_Gaze_MLP', 'AdaGTCN_lite']
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
    summary_path = os.path.join(RESULTS_DIR, "fewshot_gcn_proxy_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("GCN SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("\nDONE!")

if __name__ == '__main__':
    main()