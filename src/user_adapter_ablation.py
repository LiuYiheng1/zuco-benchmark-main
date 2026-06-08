"""
User Adapter: Lightweight Adaptation for EEG-based User Calibration

This module implements various lightweight adaptation strategies for personalized EEG classification.

Models:
1. EEG_MLP_baseline - Simple MLP without pre-training
2. SIED_encoder_linear_probe - Freeze SIED encoder, train only classifier
3. SIED_encoder_finetune - Fine-tune full encoder + classifier
4. SIED_encoder_adapter - Add lightweight residual adapter
5. SIED_encoder_bias_calibration - Only update classifier head and layer norms

Success criteria:
- Beat EEG_MLP by ≥2% on average at 3/5/10-shot
- OR achieve same performance with fewer calibration samples
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from copy import deepcopy

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
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)

class TaskClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class ResidualAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.up = nn.Linear(dim // reduction, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        return self.norm(residual + 0.1 * x)

class SubjectDiscriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def train_sied_encoder(X_train, y_train, sub_ids_train, n_subjects, device, lambda_adv=0.01, epochs=30):
    """Train SIED adversarial encoder"""
    eeg_dim = X_train.shape[1]
    encoder = EEGEncoder(eeg_dim).to(device)
    sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    sub_tr_t = torch.LongTensor(sub_ids_train).to(device)

    for epoch in range(epochs):
        encoder.train()
        sub_disc.train()

        z = encoder(X_tr_t)
        reversed_z = GradientReversalFunction.apply(z, lambda_adv)
        sub_logits = sub_disc(reversed_z)
        sub_loss = F.cross_entropy(sub_logits, sub_tr_t)

        optimizer.zero_grad()
        sub_loss.backward()
        optimizer.step()

    encoder.eval()
    return encoder

def run_eeg_mlp_baseline(X_cal, y_cal, X_test, y_test):
    """EEG_MLP_baseline: Simple MLP without pre-training"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

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

def run_sied_linear_probe(sied_encoder, X_cal, y_cal, X_test, y_test, device):
    """SIED_encoder_linear_probe: Freeze encoder, train only classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    with torch.no_grad():
        z_cal = sied_encoder(torch.FloatTensor(X_cal_s).to(device)).cpu().numpy()
        z_test = sied_encoder(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(z_cal, y_cal)
    preds = clf.predict(z_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(z_test)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_sied_finetune(sied_encoder, X_cal, y_cal, X_test, y_test, device):
    """SIED_encoder_finetune: Fine-tune encoder + classifier with small LR"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    encoder = deepcopy(sied_encoder).to(device)
    clf_head = TaskClassifier(encoder.output_dim).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(clf_head.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_t = torch.FloatTensor(X_cal_s).to(device)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)

    encoder.train()
    clf_head.train()

    for epoch in range(50):
        z = encoder(X_cal_t)
        logits = clf_head(z)
        loss = criterion(logits, y_cal_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    clf_head.eval()

    with torch.no_grad():
        z_test = encoder(torch.FloatTensor(X_test_s).to(device))
        test_logits = clf_head(z_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    bacc = balanced_accuracy_score(y_test, test_preds)
    try:
        auroc = roc_auc_score(y_test, test_probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_sied_adapter(sied_encoder, X_cal, y_cal, X_test, y_test, device):
    """SIED_encoder_adapter: Add lightweight residual adapter"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    encoder = deepcopy(sied_encoder).to(device)
    adapter = ResidualAdapter(encoder.output_dim).to(device)
    clf_head = TaskClassifier(encoder.output_dim).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(adapter.parameters()) + list(clf_head.parameters()),
                         lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_t = torch.FloatTensor(X_cal_s).to(device)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)

    encoder.train()
    adapter.train()
    clf_head.train()

    for epoch in range(50):
        z = encoder(X_cal_t)
        z = adapter(z)
        logits = clf_head(z)
        loss = criterion(logits, y_cal_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    adapter.eval()
    clf_head.eval()

    with torch.no_grad():
        z_test = encoder(torch.FloatTensor(X_test_s).to(device))
        z_test = adapter(z_test)
        test_logits = clf_head(z_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    bacc = balanced_accuracy_score(y_test, test_preds)
    try:
        auroc = roc_auc_score(y_test, test_probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_sied_bias_calibration(sied_encoder, X_cal, y_cal, X_test, y_test, device):
    """SIED_encoder_bias_calibration: Only update classifier head (bias-only adaptation)"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    encoder = deepcopy(sied_encoder).to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    clf_head = nn.Sequential(
        nn.Linear(encoder.output_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    ).to(device)

    optimizer = optim.Adam(clf_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_cal_t = torch.FloatTensor(X_cal_s).to(device)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)

    clf_head.train()

    for epoch in range(100):
        with torch.no_grad():
            z = encoder(X_cal_t)
        logits = clf_head(z)
        loss = criterion(logits, y_cal_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    clf_head.eval()

    with torch.no_grad():
        z_test = encoder(torch.FloatTensor(X_test_s).to(device))
        test_logits = clf_head(z_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    bacc = balanced_accuracy_score(y_test, test_preds)
    try:
        auroc = roc_auc_score(y_test, test_probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment(seed, model_type, sied_encoders=None):
    """Run user adapter experiment"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for held_out in Y_SUBJECTS:
        print(f"\n  {model_type} - {held_out}:", flush=True)

        X_eeg, y_eeg = load_eeg_data(held_out)
        if X_eeg is None or len(X_eeg) < 50:
            continue

        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all, sub_ids_train_all = [], [], []
        for subj_idx, subj in enumerate(train_subjs):
            X_subj, y_subj = load_eeg_data(subj)
            if X_subj is not None:
                X_train_all.append(X_subj)
                y_train_all.append(y_subj)
                sub_ids_train_all.extend([subj_idx] * len(y_subj))

        if len(X_train_all) == 0:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids_train_all = np.array(sub_ids_train_all)

        n_subjects = len(np.unique(sub_ids_train_all))

        if model_type != 'EEG_MLP_baseline' and sied_encoders is not None:
            if held_out not in sied_encoders:
                scaler_tmp = StandardScaler()
                X_train_s = scaler_tmp.fit_transform(X_train_all)
                sied_encoders[held_out] = train_sied_encoder(
                    X_train_s, y_train_all, sub_ids_train_all, n_subjects, device)

        n_samples = len(y_eeg)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_samples // 2]
        cal_pool_indices = indices[n_samples // 2:]

        X_test = X_eeg[test_indices]
        y_test = y_eeg[test_indices]
        X_cal_pool = X_eeg[cal_pool_indices]
        y_cal_pool = y_eeg[cal_pool_indices]

        for n_cal_per_class in calibration_settings:
            if n_cal_per_class * 2 > len(cal_pool_indices):
                continue

            cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
            cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
            cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
            np.random.shuffle(cal_idx)

            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if model_type == 'EEG_MLP_baseline':
                acc, f1, bacc, auroc = run_eeg_mlp_baseline(X_cal, y_cal, X_test, y_test)
            elif model_type == 'SIED_encoder_linear_probe':
                acc, f1, bacc, auroc = run_sied_linear_probe(
                    sied_encoders[held_out], X_cal, y_cal, X_test, y_test, device)
            elif model_type == 'SIED_encoder_finetune':
                acc, f1, bacc, auroc = run_sied_finetune(
                    sied_encoders[held_out], X_cal, y_cal, X_test, y_test, device)
            elif model_type == 'SIED_encoder_adapter':
                acc, f1, bacc, auroc = run_sied_adapter(
                    sied_encoders[held_out], X_cal, y_cal, X_test, y_test, device)
            elif model_type == 'SIED_encoder_bias_calibration':
                acc, f1, bacc, auroc = run_sied_bias_calibration(
                    sied_encoders[held_out], X_cal, y_cal, X_test, y_test, device)
            else:
                continue

            results.append({
                'model': model_type,
                'seed': seed,
                'subject': held_out,
                'n_cal_per_class': n_cal_per_class,
                'n_cal_total': n_cal_per_class * 2,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'auroc': auroc
            })

            print(f"    {n_cal_per_class}-shot: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}", flush=True)

    return results, sied_encoders

def main():
    print("="*70)
    print("User Adapter Ablation Experiment")
    print("="*70)

    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []

    model_types = [
        'EEG_MLP_baseline',
        'SIED_encoder_linear_probe',
        'SIED_encoder_finetune',
        'SIED_encoder_adapter',
        'SIED_encoder_bias_calibration'
    ]

    sied_encoders = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            print(f"\n--- Seed {seed} ---", flush=True)
            results, sied_encoders = run_experiment(seed, model_type, sied_encoders)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "user_adapter_ablation.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "user_adapter_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()