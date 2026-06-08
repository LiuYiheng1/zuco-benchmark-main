"""
TCD: Task-Confound Disentanglement

Based on "Disentangled Representation Learning for Robust Brainprint Recognition"
Adapted for NR/TSR reading state classification

Models:
1. SIED - subject-invariant adversarial (baseline)
2. TCD_no_conf_branch - single branch (no disentanglement)
3. TCD_no_corr - without correlation constraint
4. TCD_no_recon - without reconstruction loss
5. TCD_full - full model with task/confound branches
6. TCD_full_plus_TaskSupCon - full model with SupCon on task embeddings

Protocol: LOSO cross-subject (16 subjects)
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
from sklearn.linear_model import SGDClassifier
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/domain_generalization"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class TaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)

class ConfoundEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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

class SubjectClassifier(nn.Module):
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

class SubjectAdvClassifier(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            GradientReversalLayer(1.0),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, concat_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

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

def correlation_loss(z1, z2):
    """Minimize correlation between two embedding sets"""
    z1 = z1 - z1.mean(dim=1, keepdim=True)
    z2 = z2 - z2.mean(dim=1, keepdim=True)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    corr = (z1 * z2).sum(dim=1).mean()
    return corr ** 2

def supervised_contrastive_loss(z, y, temperature=0.1):
    """SupCon on task embeddings only (same label = positive)"""
    device = z.device
    batch_size = z.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    y = y.view(-1)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature

    mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float().to(device)
    mask_no_self = mask - torch.eye(batch_size).to(device)
    pos_mask = mask_no_self
    neg_mask = 1.0 - mask

    pos_sim = (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    neg_sim = (sim_matrix * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)

    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8))
    return loss.mean()

def run_sied(seed, lambda_adv=1.0):
    """SIED baseline"""
    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all, sub_ids = [], [], []

        for subj_idx, subj in enumerate(train_subjs):
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)
                sub_ids.extend([subj_idx] * len(y))

        X_test, y_test = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids = np.array(sub_ids)

        np.random.seed(seed)
        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        y_tr = y_train_all[train_idx]
        sub_tr = sub_ids[train_idx]
        X_val = scaler.transform(X_train_all[val_idx])
        y_val = y_train_all[val_idx]
        X_test_s = scaler.transform(X_test)

        eeg_dim = X_tr.shape[1]
        encoder = TaskEncoder(eeg_dim).to(device)
        task_clf = TaskClassifier(encoder.output_dim).to(device)
        sub_adv = SubjectAdvClassifier(encoder.output_dim, n_subjects).to(device)

        optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_adv.parameters()), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        sub_tr_t = torch.LongTensor(sub_tr).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        best_val_f1 = 0
        best_encoder_state = None
        best_clf_state = None
        patience_counter = 0

        for epoch in range(30):
            encoder.train()
            task_clf.train()
            sub_adv.train()

            z = encoder(X_tr_t)
            task_logits = task_clf(z)
            sub_logits = sub_adv(z)

            task_loss = criterion(task_logits, y_tr_t)
            sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
            loss = task_loss + lambda_adv * sub_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            encoder.eval()
            task_clf.eval()
            with torch.no_grad():
                val_z = encoder(X_val_t)
                val_logits = task_clf(val_z)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_f1 = f1_score(y_val, val_preds.cpu().numpy(), average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_encoder_state = encoder.state_dict().copy()
                best_clf_state = task_clf.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state)
            task_clf.load_state_dict(best_clf_state)

        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z = encoder(X_test_t)
            test_logits = task_clf(test_z)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        bacc = balanced_accuracy_score(y_test, test_preds)
        try:
            auroc = roc_auc_score(y_test, test_probs)
        except:
            auroc = 0.5

        results.append({
            'model': 'SIED', 'seed': seed, 'held_out': held_out,
            'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
        })

    return results

def run_tcd_full(seed, lambda_adv=1.0, lambda_conf=0.5, lambda_corr=0.1, lambda_recon=0.1, lambda_supcon=0.0, temperature=0.1):
    """TCD full model with all components"""
    model_name = 'TCD_full'
    if lambda_supcon > 0:
        model_name = 'TCD_full_plus_SupCon'

    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all, sub_ids = [], [], []

        for subj_idx, subj in enumerate(train_subjs):
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)
                sub_ids.extend([subj_idx] * len(y))

        X_test, y_test = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids = np.array(sub_ids)

        np.random.seed(seed)
        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        y_tr = y_train_all[train_idx]
        sub_tr = sub_ids[train_idx]
        X_val = scaler.transform(X_train_all[val_idx])
        y_val = y_train_all[val_idx]
        X_test_s = scaler.transform(X_test)

        eeg_dim = X_tr.shape[1]

        task_enc = TaskEncoder(eeg_dim).to(device)
        conf_enc = ConfoundEncoder(eeg_dim).to(device)
        task_clf = TaskClassifier(task_enc.output_dim).to(device)
        sub_adv = SubjectAdvClassifier(task_enc.output_dim, n_subjects).to(device)
        conf_clf = SubjectClassifier(conf_enc.output_dim, n_subjects).to(device)
        decoder = Decoder(task_enc.output_dim + conf_enc.output_dim, eeg_dim).to(device)

        all_params = list(task_enc.parameters()) + list(conf_enc.parameters()) + list(task_clf.parameters()) + list(sub_adv.parameters()) + list(conf_clf.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(all_params, lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        y_tr_labels = torch.LongTensor(y_tr).to(device)
        sub_tr_t = torch.LongTensor(sub_tr).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        best_val_f1 = 0
        best_states = None
        patience_counter = 0

        for epoch in range(30):
            task_enc.train()
            conf_enc.train()
            task_clf.train()
            sub_adv.train()
            conf_clf.train()
            decoder.train()

            z_task = task_enc(X_tr_t)
            z_conf = conf_enc(X_tr_t)

            task_logits = task_clf(z_task)
            sub_logits = sub_adv(z_task)
            conf_logits = conf_clf(z_conf)

            task_loss = criterion(task_logits, y_tr_t)
            sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
            conf_loss = F.cross_entropy(conf_logits, sub_tr_t)

            corr_loss_val = correlation_loss(z_task, z_conf) if lambda_corr > 0 else torch.tensor(0.0)

            recon = decoder(torch.cat([z_task, z_conf], dim=1))
            recon_loss = F.mse_loss(recon, X_tr_t) if lambda_recon > 0 else torch.tensor(0.0)

            supcon_loss = supervised_contrastive_loss(z_task, y_tr_labels, temperature) if lambda_supcon > 0 else torch.tensor(0.0)

            loss = task_loss + lambda_adv * sub_loss + lambda_conf * conf_loss + lambda_corr * corr_loss_val + lambda_recon * recon_loss + lambda_supcon * supcon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            task_enc.eval()
            task_clf.eval()
            with torch.no_grad():
                val_z_task = task_enc(X_val_t)
                val_logits = task_clf(val_z_task)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_f1 = f1_score(y_val, val_preds.cpu().numpy(), average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_states = {
                    'task_enc': task_enc.state_dict().copy(),
                    'conf_enc': conf_enc.state_dict().copy(),
                    'task_clf': task_clf.state_dict().copy(),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

        if best_states is not None:
            task_enc.load_state_dict(best_states['task_enc'])
            conf_enc.load_state_dict(best_states['conf_enc'])
            task_clf.load_state_dict(best_states['task_clf'])

        task_enc.eval()
        conf_enc.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z_task = task_enc(X_test_t)
            test_logits = task_clf(test_z_task)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        bacc = balanced_accuracy_score(y_test, test_preds)
        try:
            auroc = roc_auc_score(y_test, test_probs)
        except:
            auroc = 0.5

        results.append({
            'model': model_name, 'seed': seed, 'held_out': held_out,
            'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
        })

    return results

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []

    print("Running SIED baseline...", flush=True)
    all_results.extend(run_sied(seed=0))

    print("Running TCD_full...", flush=True)
    all_results.extend(run_tcd_full(seed=0, lambda_adv=1.0, lambda_conf=0.5, lambda_corr=0.1, lambda_recon=0.1, lambda_supcon=0.0))

    print("Running TCD_full_plus_SupCon...", flush=True)
    all_results.extend(run_tcd_full(seed=0, lambda_adv=1.0, lambda_conf=0.5, lambda_corr=0.1, lambda_recon=0.1, lambda_supcon=0.1, temperature=0.1))

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "tcd_results.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*60, flush=True)
    print("Results Summary", flush=True)
    print("="*60, flush=True)

    for model in df['model'].unique():
        data = df[df['model'] == model]
        acc = data['accuracy'].mean()
        std = data['accuracy'].std()
        f1 = data['macro_f1'].mean()
        print(f"  {model:30s}: acc={acc:.4f}±{std:.4f}, f1={f1:.4f}", flush=True)

    sied_baseline = df[df['model'] == 'SIED']['accuracy'].mean()
    print(f"\nSIED baseline: {sied_baseline:.4f}", flush=True)
    print(f"Target (SIED + 1.5%): {sied_baseline + 0.015:.4f}", flush=True)
    print(f"Target (SIED + 2.0%): {sied_baseline + 0.02:.4f}", flush=True)

    print(f"\nResults saved to {output_path}", flush=True)

if __name__ == '__main__':
    main()