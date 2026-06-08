"""
SS-CMC: Semi-Supervised Cross-Modal Consistency Calibration

Goal: Leverage unlabeled EEG/Gaze data to improve personalized models and fusion.

Architecture:
- Two branch: EEG encoder + Gaze encoder
- Supervised loss on labeled calibration samples
- Consistency loss: KL divergence between EEG and Gaze predictions on unlabeled samples
- Confidence threshold to filter unreliable unlabeled samples
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
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

def load_gaze_data(subject):
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
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)

class GazeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_ss_cmc(X_eeg_cal, y_cal, X_gaze_cal,
                 X_eeg_unlab, X_gaze_unlab,
                 X_eeg_test, X_gaze_test, y_test,
                 lambda_cons=0.1, tau=0.7, device='cpu'):
    """Train SS-CMC model"""

    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()

    X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
    X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
    X_eeg_unlab_s = scaler_eeg.transform(X_eeg_unlab)
    X_gaze_unlab_s = scaler_gaze.transform(X_gaze_unlab)
    X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

    eeg_dim = X_eeg_cal_s.shape[1]
    gaze_dim = X_gaze_cal_s.shape[1]

    encoder_eeg = EEGEncoder(eeg_dim).to(device)
    encoder_gaze = GazeEncoder(gaze_dim).to(device)
    clf_eeg = ClassifierHead(64).to(device)
    clf_gaze = ClassifierHead(64).to(device)
    clf_fusion = ClassifierHead(128).to(device)

    optimizer = optim.Adam(
        list(encoder_eeg.parameters()) + list(encoder_gaze.parameters()) +
        list(clf_eeg.parameters()) + list(clf_gaze.parameters()) + list(clf_fusion.parameters()),
        lr=0.001, weight_decay=1e-4
    )

    X_cal_eeg_t = torch.FloatTensor(X_eeg_cal_s).to(device)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)
    X_cal_gaze_t = torch.FloatTensor(X_gaze_cal_s).to(device)
    X_unlab_eeg_t = torch.FloatTensor(X_eeg_unlab_s).to(device)
    X_unlab_gaze_t = torch.FloatTensor(X_gaze_unlab_s).to(device)

    for epoch in range(50):
        encoder_eeg.train()
        encoder_gaze.train()
        clf_eeg.train()
        clf_gaze.train()
        clf_fusion.train()

        z_eeg_cal = encoder_eeg(X_cal_eeg_t)
        z_gaze_cal = encoder_gaze(X_cal_gaze_t)

        logits_eeg = clf_eeg(z_eeg_cal)
        logits_gaze = clf_gaze(z_gaze_cal)

        z_fusion = torch.cat([z_eeg_cal, z_gaze_cal], dim=1)
        logits_fusion = clf_fusion(z_fusion)

        loss_sup = F.binary_cross_entropy_with_logits(logits_eeg, y_cal_t) + \
                   F.binary_cross_entropy_with_logits(logits_gaze, y_cal_t) + \
                   F.binary_cross_entropy_with_logits(logits_fusion, y_cal_t)

        if len(X_unlab_eeg_t) > 0:
            encoder_eeg.eval()
            encoder_gaze.eval()
            with torch.no_grad():
                z_eeg_unlab = encoder_eeg(X_unlab_eeg_t)
                z_gaze_unlab = encoder_gaze(X_unlab_gaze_t)

            probs_eeg = torch.sigmoid(clf_eeg(z_eeg_unlab))
            probs_gaze = torch.sigmoid(clf_gaze(z_gaze_unlab))

            max_probs, _ = torch.max(torch.cat([probs_eeg, 1-probs_eeg, probs_gaze, 1-probs_gaze], dim=1), dim=1)
            mask = (max_probs > tau).float()

            if mask.sum() > 0:
                p_eeg = torch.cat([probs_eeg, 1-probs_eeg], dim=1)
                p_gaze = torch.cat([probs_gaze, 1-probs_gaze], dim=1)
                p_eeg = p_eeg / (p_eeg.sum(dim=1, keepdim=True) + 1e-8)
                p_gaze = p_gaze / (p_gaze.sum(dim=1, keepdim=True) + 1e-8)

                loss_cons = F.kl_div(p_eeg.log(), p_gaze.detach(), reduction='none').sum(dim=1)
                loss_cons = (loss_cons * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss_cons = torch.tensor(0.0).to(device)
        else:
            loss_cons = torch.tensor(0.0).to(device)

        loss = loss_sup + lambda_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder_eeg.eval()
    encoder_gaze.eval()
    clf_eeg.eval()
    clf_gaze.eval()
    clf_fusion.eval()

    with torch.no_grad():
        z_eeg_test = encoder_eeg(torch.FloatTensor(X_eeg_test_s).to(device))
        z_gaze_test = encoder_gaze(torch.FloatTensor(X_gaze_test_s).to(device))
        z_fusion_test = torch.cat([z_eeg_test, z_gaze_test], dim=1)

        logits_fusion = clf_fusion(z_fusion_test)
        probs = torch.sigmoid(logits_fusion).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def train_baseline(X_cal, y_cal, X_test, y_test, model_type='EEG_MLP'):
    """Train baseline classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    if model_type == 'EEG_SVM':
        clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    clf.fit(X_cal_s, y_cal)
    preds = clf.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def train_static_fusion(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test, y_test):
    """Train static EEG-Gaze fusion as baseline"""
    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()

    X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
    X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
    X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

    clf_eeg = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf_gaze = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)

    clf_eeg.fit(X_eeg_cal_s, y_cal)
    clf_gaze.fit(X_gaze_cal_s, y_cal)

    prob_eeg = clf_eeg.predict_proba(X_eeg_test_s)[:, 1]
    prob_gaze = clf_gaze.predict_proba(X_gaze_test_s)[:, 1]

    prob_fusion = 0.5 * prob_eeg + 0.5 * prob_gaze
    preds = (prob_fusion > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, prob_fusion)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]
    lambda_cons_values = [0.01, 0.05, 0.1]
    tau_values = [0.7, 0.8]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("SS-CMC: Semi-Supervised Cross-Modal Consistency")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:")

        for held_out in Y_SUBJECTS:
            X_eeg, y_eeg = load_eeg_data(held_out)
            X_gaze, y_gaze = load_gaze_data(held_out)

            if X_eeg is None or X_gaze is None:
                continue

            common_len = min(len(y_eeg), len(y_gaze))
            X_eeg = X_eeg[:common_len]
            y_eeg = y_eeg[:common_len]
            X_gaze = X_gaze[:common_len]

            if len(X_eeg) < 50:
                continue

            n_samples = len(y_eeg)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            X_test_eeg = X_eeg[test_indices]
            X_test_gaze = X_gaze[test_indices]
            y_test = y_eeg[test_indices]

            X_cal_pool_eeg = X_eeg[cal_pool_indices]
            X_cal_pool_gaze = X_gaze[cal_pool_indices]
            y_cal_pool = y_eeg[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
                cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
                cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
                np.random.shuffle(cal_idx)

                X_cal_eeg = X_cal_pool_eeg[cal_idx]
                X_cal_gaze = X_cal_pool_gaze[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                unlab_indices = np.setdiff1d(np.arange(len(cal_pool_indices)), cal_idx)
                np.random.shuffle(unlab_indices)
                n_unlab = min(len(unlab_indices), n_cal_per_class * 4)
                unlab_idx = unlab_indices[:n_unlab]

                X_unlab_eeg = X_cal_pool_eeg[unlab_idx]
                X_unlab_gaze = X_cal_pool_gaze[unlab_idx]

                eeg_acc, eeg_f1, eeg_bacc, eeg_auroc = train_baseline(X_cal_eeg, y_cal, X_test_eeg, y_test)
                gaze_acc, gaze_f1, gaze_bacc, gaze_auroc = train_baseline(X_cal_gaze, y_cal, X_test_gaze, y_test)
                fusion_acc, fusion_f1, fusion_bacc, fusion_auroc = train_static_fusion(
                    X_cal_eeg, y_cal, X_cal_gaze, X_test_eeg, X_test_gaze, y_test)

                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'method': 'EEG_only',
                    'accuracy': eeg_acc,
                    'macro_f1': eeg_f1,
                    'balanced_accuracy': eeg_bacc,
                    'auroc': eeg_auroc
                })

                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'method': 'Gaze_only',
                    'accuracy': gaze_acc,
                    'macro_f1': gaze_f1,
                    'balanced_accuracy': gaze_bacc,
                    'auroc': gaze_auroc
                })

                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'method': 'Static_EEG_Gaze_average',
                    'accuracy': fusion_acc,
                    'macro_f1': fusion_f1,
                    'balanced_accuracy': fusion_bacc,
                    'auroc': fusion_auroc
                })

                for lambda_cons in lambda_cons_values:
                    for tau in tau_values:
                        ss_cmc_acc, ss_cmc_f1, ss_cmc_bacc, ss_cmc_auroc = train_ss_cmc(
                            X_cal_eeg, y_cal, X_cal_gaze,
                            X_unlab_eeg, X_unlab_gaze,
                            X_test_eeg, X_test_gaze, y_test,
                            lambda_cons=lambda_cons, tau=tau, device=device
                        )

                        results.append({
                            'seed': seed,
                            'subject': held_out,
                            'n_cal_per_class': n_cal_per_class,
                            'n_cal_total': n_cal_per_class * 2,
                            'method': f'SS_CMC_l{lambda_cons}_t{tau}',
                            'accuracy': ss_cmc_acc,
                            'macro_f1': ss_cmc_f1,
                            'balanced_accuracy': ss_cmc_bacc,
                            'auroc': ss_cmc_auroc
                        })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "ss_cmc_results.csv")
    df.to_csv(output_path, index=False)

    summary = df.groupby(['method', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "ss_cmc_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("SUMMARY (Accuracy at 50-shot)")
    print("="*60)
    shot_50 = summary[summary['n_cal_per_class'] == 50]
    for _, row in shot_50.iterrows():
        method = row['method']
        acc = row[('accuracy', 'mean')]
        std = row[('accuracy', 'std')]
        print(f"{method}: {acc:.4f}±{std:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()