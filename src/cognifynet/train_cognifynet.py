"""
CognifyNet Training Pipeline
=============================
- Strict LOSO protocol
- Train-only graph construction
- Multi-loss training: CE + adversarial + orthogonality + supcon + energy

Losses:
  L_task    = CE(task_logits, y)
  L_subj    = CE(subj_logits, s)
  L_adv     = CE(adv_subj_logits, s)
  L_ortho   = |z_task^T @ z_subj|_F
  L_supcon  = supervised contrastive on z_task
  L_energy  = modality energy alignment
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                              roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

from .cognifynet_model import CognifyNet, EEGNeuroGraphEncoder, GazeEncoder, TaskDisentangler, EnergyFusion

OUTPUT_DIR = "results/cognifynet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = "data/aligned_multimodal_y.npz"
METADATA_FILE = "data/aligned_multimodal_y_metadata.csv"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS',
              'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

N_BANDS = 4
N_CHANNELS = 105


def load_data():
    data = np.load(DATA_FILE)
    meta = pd.read_csv(METADATA_FILE)
    X_eeg = data['eeg']
    X_gaze = data['gaze']
    y = data['y'].astype(np.int64)
    subjects = meta['subject'].values
    return X_eeg, X_gaze, y, subjects


def build_graph_adjacency(X_eeg_train, n_bands=4, n_channels=105, eps=0.3):
    N = X_eeg_train.shape[0]
    X = X_eeg_train.reshape(N, n_channels, n_bands)
    corr_matrices = []
    for b in range(n_bands):
        corr = np.corrcoef(X[:, :, b].T)
        corr = np.nan_to_num(corr, 0.0)
        corr_matrices.append(corr)
    adj = np.mean(corr_matrices, axis=0)
    adj = np.abs(adj)
    adj[adj < eps] = 0
    adj = adj + np.eye(n_channels) * 0.5
    return torch.tensor(adj, dtype=torch.float32).to(DEVICE)


def supervised_contrastive_loss(z, y, temperature=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = y.unsqueeze(0) == y.unsqueeze(1)
    sim_exp = torch.exp(sim)
    pos_sum = (sim_exp * mask.float()).sum(dim=1)
    all_sum = sim_exp.sum(dim=1) - torch.exp(torch.diag(sim))
    loss = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
    return loss


def energy_alignment_loss(energies):
    E_eeg, E_gaze, E_task = energies
    L_eeg_gaze = F.mse_loss(E_eeg, E_gaze)
    L_eeg_task = F.mse_loss(E_eeg, E_task)
    L_gaze_task = F.mse_loss(E_gaze, E_task)
    return (L_eeg_gaze + L_eeg_task + L_gaze_task) / 3


def compute_loss(task_logits, z_task, z_subj, z_task_grl, w, energies,
                 y, s, subj_logits, adv_subj_logits,
                 lambda_adv=0.1, lambda_ortho=0.05, lambda_supcon=0.1, lambda_energy=0.05):
    L_task = F.cross_entropy(task_logits, y)
    L_subj = F.cross_entropy(subj_logits, s)
    L_adv = F.cross_entropy(adv_subj_logits, s)
    L_ortho = (z_task.T @ z_subj).abs().mean()
    L_supcon = supervised_contrastive_loss(z_task, y, temperature=0.1)
    L_energy = energy_alignment_loss(energies)

    L_total = (L_task +
               L_subj +
               lambda_adv * L_adv +
               lambda_ortho * L_ortho +
               lambda_supcon * L_supcon +
               lambda_energy * L_energy)
    return L_total


def train_one_fold(X_train, X_test, y_train, y_test, subjects_train, subjects_test,
                   model_type='full', epochs=80, batch_size=64, lr=1e-3):
    le = LabelEncoder()
    s_train = le.fit_transform(subjects_train)
    n_subjects = len(le.classes_)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    eeg_scaler = StandardScaler()
    gaze_scaler = StandardScaler()
    X_eeg_train = eeg_scaler.fit_transform(X_train[:, :420])
    X_gaze_train = gaze_scaler.fit_transform(X_train[:, 420:])
    X_eeg_test = eeg_scaler.transform(X_test[:, :420])
    X_gaze_test = gaze_scaler.transform(X_test[:, 420:])

    adj = build_graph_adjacency(X_eeg_train, N_BANDS, N_CHANNELS)

    train_ds = TensorDataset(
        torch.tensor(X_eeg_train, dtype=torch.float32),
        torch.tensor(X_gaze_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(s_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = CognifyNet(
        n_bands=N_BANDS, n_channels=N_CHANNELS,
        eeg_hidden=128, gcn_hidden=256,
        gaze_hidden=64, gaze_dim=128,
        task_dim=128, subj_dim=128,
        n_classes=2, dropout=0.3,
    ).to(DEVICE)
    model.set_subj_head(n_subjects)
    model.subj_head = model.subj_head.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_eeg, x_gz, yb, sb in train_loader:
            x_eeg, x_gz, yb, sb = x_eeg.to(DEVICE), x_gz.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            optimizer.zero_grad()
            task_logits, z_task, z_subj, z_task_grl, w, energies = model(x_eeg, x_gz, adj, lamda=0.1)
            subj_logits = model.subj_head(z_subj)
            adv_subj_logits = model.subj_head(z_task_grl)
            loss = compute_loss(task_logits, z_task, z_subj, z_task_grl, w, energies, yb, sb, subj_logits, adv_subj_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        x_eeg_t = torch.tensor(X_eeg_test, dtype=torch.float32).to(DEVICE)
        x_gz_t = torch.tensor(X_gaze_test, dtype=torch.float32).to(DEVICE)
        task_logits, _, _, _, w, _ = model(x_eeg_t, x_gz_t, adj, lamda=0.1)
        y_proba = F.softmax(task_logits, dim=-1).cpu().numpy()[:, 1]
        y_pred = task_logits.argmax(dim=-1).cpu().numpy()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'auroc': roc_auc_score(y_test, y_proba),
    }


def run_loso_evaluation(X_concat, y, subjects, model_type='full', epochs=80):
    results = []
    for test_subj in Y_SUBJECTS:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        X_train = X_concat[train_mask]
        X_test = X_concat[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        subj_train = subjects[train_mask]
        subj_test = subjects[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        metrics = train_one_fold(X_train, X_test, y_train, y_test,
                                  subj_train, subj_test, model_type=model_type, epochs=epochs)
        metrics['test_subject'] = test_subj
        metrics['test_N'] = len(y_test)
        results.append(metrics)
        print(f"  {test_subj}: Macro-F1={metrics['macro_f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return pd.DataFrame(results)


def smoke_test():
    print("=" * 70)
    print("CognifyNet Smoke Test (3 subjects)")
    print("=" * 70)
    X_eeg, X_gaze, y, subjects = load_data()
    X_concat = np.hstack([X_eeg, X_gaze])

    smoke_subjects = ['YHS', 'YRK', 'YFR']
    results = []

    for test_subj in smoke_subjects:
        print(f"\n  --- Fold: test={test_subj} ---")
        train_mask = np.array([s not in [test_subj] for s in subjects])
        test_mask = subjects == test_subj

        X_train = X_concat[train_mask]
        X_test = X_concat[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        subj_train = subjects[train_mask]
        subj_test = subjects[test_mask]

        metrics = train_one_fold(X_train, X_test, y_train, y_test,
                                  subj_train, subj_test, model_type='full', epochs=10)
        metrics['test_subject'] = test_subj
        metrics['test_N'] = len(y_test)
        results.append(metrics)
        print(f"  {test_subj}: Macro-F1={metrics['macro_f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    df = pd.DataFrame(results)
    print(f"\n  Mean Macro-F1: {df['macro_f1'].mean():.4f} +/- {df['macro_f1'].std():.4f}")
    df.to_csv(os.path.join(OUTPUT_DIR, "smoke_test_results.csv"), index=False)
    return df


def full_loso():
    print("=" * 70)
    print("CognifyNet Full LOSO (16 subjects)")
    print("=" * 70)
    X_eeg, X_gaze, y, subjects = load_data()
    X_concat = np.hstack([X_eeg, X_gaze])

    df = run_loso_evaluation(X_concat, y, subjects, model_type='full', epochs=60)
    print(f"\n  {'='*50}")
    print(f"  Mean Macro-F1: {df['macro_f1'].mean():.4f} +/- {df['macro_f1'].std():.4f}")
    print(f"  Mean Accuracy: {df['accuracy'].mean():.4f} +/- {df['accuracy'].std():.4f}")
    print(f"  Mean AUROC: {df['auroc'].mean():.4f} +/- {df['auroc'].std():.4f}")
    df.to_csv(os.path.join(OUTPUT_DIR, "full_loso_results.csv"), index=False)
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("CognifyNet Training Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df_smoke = smoke_test()

    smoke_f1 = df_smoke['macro_f1'].mean()
    if smoke_f1 >= 0.55:
        print(f"\n  Smoke test PASS (Macro-F1={smoke_f1:.4f} >= 0.55)")
        print("  Running full LOSO...")
        df_full = full_loso()
    else:
        print(f"\n  Smoke test FAIL (Macro-F1={smoke_f1:.4f} < 0.55)")
        print("  Stopping, not running full LOSO.")

    print(f"\n  Results saved to: {OUTPUT_DIR}/")