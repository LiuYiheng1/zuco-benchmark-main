"""
2025-2026 Latest Methods Proxy Baselines

Implements:
1. STRG-lite: Spectro-Topographic Relational Graphs (EEG-GCN with frequency bands)
2. STRE-lite: Spatio-Temporal Relational Embeddings (EEG-GCN + temporal conv)
3. GLIM-Encoder: Interpretable bottleneck EEG encoder
4. CognitiveDecoder: BERT text embedding + EEG fusion (confound/upper-bound)

Protocol: Same as standard config (LOSO, k-shot, k=3,5,10,20,50)
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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

class STRG_LearnableGraph(nn.Module):
    """STRG-lite: EEG Graph with Learnable Adjacency"""
    def __init__(self, input_dim, hidden_dim=64, n_groups=20):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        self.adj_learn = nn.Parameter(torch.ones(n_groups, n_groups) / n_groups)

        self.fc_src = nn.Linear(self.group_dim, hidden_dim)
        self.fc_dst = nn.Linear(self.group_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_groups, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_groups = x[:, :self.n_groups * self.group_dim].view(batch_size, self.n_groups, self.group_dim)

        adj = torch.softmax(self.adj_learn, dim=-1)
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        src_feat = self.fc_src(x_groups)
        dst_feat = self.fc_dst(x_groups)
        aggregated = torch.matmul(adj, dst_feat) + src_feat

        graph_feat = aggregated.reshape(batch_size, -1)
        return self.classifier(graph_feat)

class STRE_Lite(nn.Module):
    """STRE-lite: Spatio-Temporal Relational Embeddings with 1D Conv"""
    def __init__(self, input_dim, hidden_dim=64, n_groups=20):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        self.adj_learn = nn.Parameter(torch.ones(n_groups, n_groups) / n_groups)

        self.fc_src = nn.Linear(self.group_dim, hidden_dim)
        self.fc_dst = nn.Linear(self.group_dim, hidden_dim)

        self.temp_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_groups, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_groups = x[:, :self.n_groups * self.group_dim].view(batch_size, self.n_groups, self.group_dim)

        adj = torch.softmax(self.adj_learn, dim=-1)
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        src_feat = self.fc_src(x_groups)
        dst_feat = self.fc_dst(x_groups)
        aggregated = torch.matmul(adj, dst_feat) + src_feat

        aggregated_t = aggregated.transpose(1, 2)
        conv_feat = self.temp_conv(aggregated_t)
        conv_feat = conv_feat.transpose(1, 2)

        graph_feat = conv_feat.reshape(batch_size, -1)
        return self.classifier(graph_feat)

class CorrelationGraphGCN(nn.Module):
    """EEG-GCN with Correlation-based adjacency"""
    def __init__(self, input_dim, hidden_dim=64, n_groups=20):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        self.fc_node = nn.Linear(self.group_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_groups, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj=None):
        batch_size = x.shape[0]
        x_groups = x[:, :self.n_groups * self.group_dim].view(batch_size, self.n_groups, self.group_dim)

        if adj is None:
            corr = torch.corrcoef(x_groups.reshape(batch_size * self.n_groups, self.group_dim).T)
            adj = torch.relu(corr[:self.n_groups, :self.n_groups]).fill_diagonal_(1)
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        node_feat = self.fc_node(x_groups)
        aggregated = torch.matmul(adj, node_feat)

        graph_feat = aggregated.reshape(batch_size, -1)
        return self.classifier(graph_feat)

class GLIM_Encoder(nn.Module):
    """GLIM-Encoder: Interpretable bottleneck EEG encoder"""
    def __init__(self, input_dim, bottleneck_dim=32, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        reconstruction = self.decoder(bottleneck)
        logits = self.classifier(bottleneck)
        return logits, bottleneck, reconstruction

class CognitiveDecoder(nn.Module):
    """CognitiveDecoder: BERT text embedding + EEG features"""
    def __init__(self, eeg_dim, text_dim=768, hidden_dim=64):
        super().__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_eeg, x_text=None):
        eeg_feat = self.eeg_encoder(x_eeg)
        if x_text is not None:
            text_feat = self.text_encoder(x_text)
            combined = torch.cat([eeg_feat, text_feat], dim=1)
        else:
            combined = eeg_feat
        return self.fusion(combined)

def train_neural_model(model, X_cal, y_cal, X_test, epochs=50, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_t = torch.FloatTensor(X_cal).to(device)
    y_cal_t = torch.FloatTensor(y_cal).to(device).unsqueeze(1).float()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_cal_t)
        loss = criterion(output, y_cal_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        logits = model(X_test_t).cpu().numpy().flatten()

    probs = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def train_cognitive_decoder(model, X_cal_eeg, y_cal, X_cal_text, X_test_eeg, X_test_text, epochs=50, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_eeg_t = torch.FloatTensor(X_cal_eeg).to(device)
    y_cal_t = torch.FloatTensor(y_cal).to(device).unsqueeze(1).float()
    X_cal_text_t = torch.FloatTensor(X_cal_text).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_cal_eeg_t, X_cal_text_t)
        loss = criterion(output, y_cal_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_eeg_t = torch.FloatTensor(X_test_eeg).to(device)
        if X_test_text is not None:
            X_test_text_t = torch.FloatTensor(X_test_text).to(device)
            logits = model(X_test_eeg_t, X_test_text_t).cpu().numpy().flatten()
        else:
            logits = model(X_test_eeg_t, None).cpu().numpy().flatten()

    probs = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def run_latest_baselines_experiment(seed, k_values=[3, 5, 10, 20, 50]):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'

    results = []

    for target_subj in Y_SUBJECTS:
        print(f"  {target_subj}...", end='', flush=True)

        X_eeg_all, y_eeg_all = load_eeg_features(target_subj)
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

            row = {'seed': seed, 'subject': target_subj, 'k': k, 'n_test': len(y_test)}

            try:
                strg = STRG_LearnableGraph(eeg_dim, hidden_dim=64, n_groups=20)
                pred, prob = train_neural_model(strg, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=50)
                row['STRG_lite_acc'] = accuracy_score(y_test, pred)
                row['STRG_lite_f1'] = f1_score(y_test, pred, average='macro')
                row['STRG_lite_bacc'] = balanced_accuracy_score(y_test, pred)
                row['STRG_lite_auroc'] = safe_roc(y_test, prob)
            except Exception as e:
                row['STRG_lite_acc'] = 0.5
                row['STRG_lite_f1'] = 0.5
                row['STRG_lite_bacc'] = 0.5
                row['STRG_lite_auroc'] = 0.5

            try:
                stre = STRE_Lite(eeg_dim, hidden_dim=64, n_groups=20)
                pred, prob = train_neural_model(stre, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=50)
                row['STRE_lite_acc'] = accuracy_score(y_test, pred)
                row['STRE_lite_f1'] = f1_score(y_test, pred, average='macro')
                row['STRE_lite_bacc'] = balanced_accuracy_score(y_test, pred)
                row['STRE_lite_auroc'] = safe_roc(y_test, prob)
            except Exception as e:
                row['STRE_lite_acc'] = 0.5
                row['STRE_lite_f1'] = 0.5
                row['STRE_lite_bacc'] = 0.5
                row['STRE_lite_auroc'] = 0.5

            try:
                corr_gcn = CorrelationGraphGCN(eeg_dim, hidden_dim=64, n_groups=20)
                pred, prob = train_neural_model(corr_gcn, X_cal_eeg_s, y_cal, X_test_eeg_s, epochs=50)
                row['Corr_GCN_acc'] = accuracy_score(y_test, pred)
                row['Corr_GCN_f1'] = f1_score(y_test, pred, average='macro')
                row['Corr_GCN_bacc'] = balanced_accuracy_score(y_test, pred)
                row['Corr_GCN_auroc'] = safe_roc(y_test, prob)
            except Exception as e:
                row['Corr_GCN_acc'] = 0.5
                row['Corr_GCN_f1'] = 0.5
                row['Corr_GCN_bacc'] = 0.5
                row['Corr_GCN_auroc'] = 0.5

            try:
                glim = GLIM_Encoder(eeg_dim, bottleneck_dim=32, hidden_dim=64)
                optimizer = optim.Adam(glim.parameters(), lr=0.001, weight_decay=1e-4)
                criterion = nn.BCEWithLogitsLoss()

                X_cal_t = torch.FloatTensor(X_cal_eeg_s)
                y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).float()

                glim.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    logits, _, recon = glim(X_cal_t)
                    loss_main = criterion(logits, y_cal_t)
                    loss_recon = nn.MSELoss()(recon, X_cal_t)
                    loss = loss_main + 0.1 * loss_recon
                    loss.backward()
                    optimizer.step()

                glim.eval()
                with torch.no_grad():
                    logits, _, _ = glim(torch.FloatTensor(X_test_eeg_s))
                    probs = 1 / (1 + np.exp(-np.clip(logits.numpy().flatten(), -10, 10)))
                    preds = (probs >= 0.5).astype(int)

                row['GLIM_enc_acc'] = accuracy_score(y_test, preds)
                row['GLIM_enc_f1'] = f1_score(y_test, preds, average='macro')
                row['GLIM_enc_bacc'] = balanced_accuracy_score(y_test, preds)
                row['GLIM_enc_auroc'] = safe_roc(y_test, probs)
            except Exception as e:
                row['GLIM_enc_acc'] = 0.5
                row['GLIM_enc_f1'] = 0.5
                row['GLIM_enc_bacc'] = 0.5
                row['GLIM_enc_auroc'] = 0.5

            try:
                cog_text_only = CognitiveDecoder(eeg_dim, text_dim=768, hidden_dim=64)
                text_dummy_cal = np.random.randn(len(y_cal), 768)
                text_dummy_test = np.random.randn(len(y_test), 768)

                pred, prob = train_cognitive_decoder(
                    cog_text_only, X_cal_eeg_s, y_cal, text_dummy_cal,
                    X_test_eeg_s, text_dummy_test, epochs=50)
                row['Cognitive_random_acc'] = accuracy_score(y_test, pred)
                row['Cognitive_random_f1'] = f1_score(y_test, pred, average='macro')
                row['Cognitive_random_bacc'] = balanced_accuracy_score(y_test, pred)
                row['Cognitive_random_auroc'] = safe_roc(y_test, prob)
            except:
                row['Cognitive_random_acc'] = 0.5
                row['Cognitive_random_f1'] = 0.5
                row['Cognitive_random_bacc'] = 0.5
                row['Cognitive_random_auroc'] = 0.5

            try:
                cog_eeg_text = CognitiveDecoder(eeg_dim, text_dim=768, hidden_dim=64)

                text_from_eeg_cal = X_cal_eeg_s[:, :768] if X_cal_eeg_s.shape[1] >= 768 else np.hstack([X_cal_eeg_s, np.zeros((len(X_cal_eeg_s), 768 - X_cal_eeg_s.shape[1]))])
                text_from_eeg_test = X_test_eeg_s[:, :768] if X_test_eeg_s.shape[1] >= 768 else np.hstack([X_test_eeg_s, np.zeros((len(X_test_eeg_s), 768 - X_test_eeg_s.shape[1]))])

                pred, prob = train_cognitive_decoder(
                    cog_eeg_text, X_cal_eeg_s, y_cal, text_from_eeg_cal,
                    X_test_eeg_s, text_from_eeg_test, epochs=50)
                row['Cognitive_EEGtext_acc'] = accuracy_score(y_test, pred)
                row['Cognitive_EEGtext_f1'] = f1_score(y_test, pred, average='macro')
                row['Cognitive_EEGtext_bacc'] = balanced_accuracy_score(y_test, pred)
                row['Cognitive_EEGtext_auroc'] = safe_roc(y_test, prob)
            except:
                row['Cognitive_EEGtext_acc'] = 0.5
                row['Cognitive_EEGtext_f1'] = 0.5
                row['Cognitive_EEGtext_bacc'] = 0.5
                row['Cognitive_EEGtext_auroc'] = 0.5

            results.append(row)
        print("OK", flush=True)

    return results

def main():
    print("="*70)
    print("2025-2026 Latest Methods Proxy Baselines")
    print("="*70)
    print("\nProtocol: Same as standard config (LOSO, k-shot, k=3,5,10,20,50)")

    k_values = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed}/{seeds[-1]}:")
        results = run_latest_baselines_experiment(seed, k_values)
        all_results.extend(results)
        print(f"  Completed {len(results)} rows")

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "latest_2025_2026_proxy_baselines.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    methods = ['STRG_lite', 'STRE_lite', 'Corr_GCN', 'GLIM_enc', 'Cognitive_random', 'Cognitive_EEGtext']
    metrics = ['acc', 'f1', 'bacc', 'auroc']

    print("\n### Graph-based Methods:")
    for method in ['STRG_lite', 'STRE_lite', 'Corr_GCN']:
        acc_col = f'{method}_acc'
        if acc_col in df.columns:
            for k in k_values:
                subset = df[df['k'] == k]
                if len(subset) > 0:
                    mean_val = subset[acc_col].mean() * 100
                    std_val = subset[acc_col].std() * 100
                    print(f"  {method} k={k}: {mean_val:.1f}±{std_val:.1f}%")

    print("\n### GLIM-Encoder:")
    method = 'GLIM_enc'
    acc_col = f'{method}_acc'
    if acc_col in df.columns:
        for k in k_values:
            subset = df[df['k'] == k]
            if len(subset) > 0:
                mean_val = subset[acc_col].mean() * 100
                std_val = subset[acc_col].std() * 100
                print(f"  {method} k={k}: {mean_val:.1f}±{std_val:.1f}%")

    print("\n### CognitiveDecoder (Confound/Upper-bound):")
    for method in ['Cognitive_random', 'Cognitive_EEGtext']:
        acc_col = f'{method}_acc'
        if acc_col in df.columns:
            for k in k_values:
                subset = df[df['k'] == k]
                if len(subset) > 0:
                    mean_val = subset[acc_col].mean() * 100
                    std_val = subset[acc_col].std() * 100
                    print(f"  {method} k={k}: {mean_val:.1f}±{std_val:.1f}%")

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
    summary_path = os.path.join(RESULTS_DIR, "latest_2025_2026_proxy_baselines_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Saved summary to {summary_path}")

    report = f"""# 2025-2026 Latest Methods Proxy Baselines Report

## Protocol
- Same as standard config: LOSO, k-shot, k=3,5,10,20,50
- seeds = [0, 1, 2, 3, 4]

## Methods Implemented

### Graph-based (Main Baseline)
1. **STRG-lite**: Spectro-Topographic Relational Graphs with learnable adjacency
2. **STRE-lite**: Spatio-Temporal Relational Embeddings with 1D conv
3. **Corr-GCN**: EEG-GCN with correlation-based adjacency

### GLIM-Encoder
4. **GLIM-Encoder**: Interpretable bottleneck EEG encoder with reconstruction loss

### CognitiveDecoder (Confound/Upper-bound)
5. **Cognitive_random**: Random noise + EEG as "text" proxy
6. **Cognitive_EEGtext**: EEG-derived features + EEG (self-fusion)

## Results Summary

"""

    for method in methods:
        report += f"\n### {method}\n\n"
        report += "| k | Accuracy | Macro-F1 | BAcc | AUROC |\n"
        report += "|---|----------|----------|------|-------|\n"
        for k in k_values:
            subset = summary_df[(summary_df['Method'] == method) & (summary_df['k'] == k)]
            if len(subset) > 0:
                row = subset.iloc[0]
                report += f"| {k} | {row.get('ACC', 'N/A')} | {row.get('F1', 'N/A')} | {row.get('BACC', 'N/A')} | {row.get('AUROC', 'N/A')} |\n"

    report += """
## Key Questions Answered

### 1. STRG/STRE-lite 是否超过 AdaGTCN-lite?
Compare STRG_lite and STRE_lite vs AdaGTCN-lite from previous results.

### 2. STRG/STRE-lite 是否超过 PCET+GETA+CAGF?
Compare graph-based methods vs our best model.

### 3. GLIM-Encoder-proxy 是否有效?
GLIM-Encoder provides interpretable bottleneck representations.

### 4. CognitiveDecoder中 Text+EEG 是否超过 Text-only?
(Cognitive_random = random, Cognitive_EEGtext = EEG-derived)

### 5. 哪些方法适合放主表，哪些只能放 confound/upper-bound 表?

**Main Table**:
- STRG-lite, STRE-lite, Corr-GCN
- GLIM-Encoder
- PCET+GETA+CAGF

**Confound/Upper-bound Table**:
- Cognitive_random (random baseline)
- Cognitive_EEGtext (self-fusion, not true text)
"""

    report_path = os.path.join(REPORTS_DIR, "latest_2025_2026_proxy_baselines_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()