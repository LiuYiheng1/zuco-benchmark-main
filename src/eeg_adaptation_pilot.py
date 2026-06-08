"""
EEG Subject-Adaptation Pilot
Tests CORAL, Adversarial, Contrastive, and GazeAnchor approaches
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from scipy.stats import pearsonr
from datetime import datetime

FEATURES_DIR = "features"
RESULTS_DIR = "results/eeg_adaptation"
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

def load_eeg_gaze_paired(subject):
    eeg_X, eeg_y = load_eeg_data(subject)
    gaze_X, gaze_y = load_gaze_data(subject)
    if eeg_X is None or gaze_X is None:
        return None, None, None, None
    common_len = min(len(eeg_X), len(gaze_X))
    return eeg_X[:common_len], gaze_X[:common_len], eeg_y[:common_len], gaze_y[:common_len]

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = hidden_dim

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

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def coral_loss(source, target, kernel='linear'):
    source = torch.FloatTensor(source)
    target = torch.FloatTensor(target)
    if source.dim() > 1:
        source = source.mean(dim=1, keepdim=True)
        target = target.mean(dim=1, keepdim=True)
    if kernel == 'linear':
        source = source - source.mean()
        target = target - target.mean()
    loss = ((source - target) ** 2).mean()
    return loss

def run_coral_experiment(seed=0):
    print("\n" + "="*60)
    print("EEG_CORAL Experiment")
    print("="*60)

    results = []
    np.random.seed(seed)

    for held_out in Y_SUBJECTS:
        print(f"\n--- Held-out: {held_out} ---")

        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_eeg_data(held_out)

        if len(X_train_all) == 0 or X_test is None or len(X_test) == 0:
            print(f"  Skipping {held_out} - no data")
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        X_tr, y_tr = X_train_all[train_idx], y_train_all[train_idx]
        X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        cov_tr = np.cov(X_tr_s.T)
        cov_test = np.cov(X_test_s.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_tr)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        d_tr = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-8)))
        whitening_tr = eigenvectors @ d_tr @ eigenvectors.T

        X_tr_coral = X_tr_s @ whitening_tr
        X_val_coral = X_val_s @ whitening_tr
        X_test_coral = X_test_s @ whitening_tr

        eigenvalues_test, eigenvectors_test = np.linalg.eigh(cov_test)
        order_test = eigenvalues_test.argsort()[::-1]
        eigenvalues_test = eigenvalues_test[order_test]
        eigenvectors_test = eigenvectors_test[:, order_test]

        d_test = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues_test, 1e-8)))
        whitening_test = eigenvectors_test @ d_test @ eigenvectors_test.T

        X_test_decor = X_test_s @ whitening_test
        X_tr_decor = X_tr_s @ whitening_test
        X_val_decor = X_val_s @ whitening_test

        alpha = 1.0
        X_tr_aligned = alpha * X_tr_coral + (1 - alpha) * X_tr_decor
        X_test_aligned = alpha * X_test_coral + (1 - alpha) * X_test_decor
        X_val_aligned = alpha * X_val_coral + (1 - alpha) * X_val_decor

        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_tr_aligned, y_tr)
        y_pred = clf.predict(X_test_aligned)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'model': 'EEG_CORAL',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'n_train': len(y_tr),
            'n_test': len(y_test)
        })
        print(f"  EEG_CORAL: Acc={acc:.4f}, F1={f1:.4f}")

    return results

def run_adversarial_experiment(seed=0, lambda_adv=0.1):
    print(f"\n--- EEG_Adversarial (lambda={lambda_adv}) ---")
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

        if len(X_train_all) == 0 or X_test is None or len(X_test) == 0:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids = np.array(sub_ids)

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
        encoder = EEGEncoder(eeg_dim).to(device)
        task_clf = TaskClassifier(encoder.output_dim).to(device)
        grl = GradientReversalLayer(lambda_adv)
        sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

        optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        sub_tr_t = torch.LongTensor(sub_tr).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        best_val_f1 = 0
        best_state = None
        patience = 0

        for epoch in range(50):
            encoder.train()
            task_clf.train()
            sub_disc.train()

            z = encoder(X_tr_t)
            task_logits = task_clf(z)
            reversed_z = grl(z)
            sub_logits = sub_disc(reversed_z)

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
                best_state = {f'encoder.{k}': v.cpu().clone() for k, v in encoder.state_dict().items()}
                best_state.update({f'task_clf.{k}': v.cpu().clone() for k, v in task_clf.state_dict().items()})
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

        if best_state is not None and len(best_state) > 0:
            encoder.load_state_dict({k.replace('encoder.', ''): v.to(device) for k, v in best_state.items() if k.startswith('encoder.')})
            task_clf.load_state_dict({k.replace('task_clf.', ''): v.to(device) for k, v in best_state.items() if k.startswith('task_clf.')})

        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z = encoder(X_test_t)
            test_logits = task_clf(test_z)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        bacc = balanced_accuracy_score(y_test, test_preds)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', warn_for=[])

        results.append({
            'model': f'EEG_Adversarial_lamb{lambda_adv}',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'n_train': len(y_tr),
            'n_test': len(y_test)
        })
        print(f"  {held_out}: Acc={acc:.4f}, F1={f1:.4f}")

    return results

def run_contrastive_experiment(seed=0, lambda_con=0.1, temperature=0.07):
    print(f"\n--- EEG_Contrastive (lambda={lambda_con}, temp={temperature}) ---")
    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []

        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_eeg_data(held_out)

        if len(X_train_all) == 0 or X_test is None or len(X_test) == 0:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        y_tr = y_train_all[train_idx]
        X_val = scaler.transform(X_train_all[val_idx])
        y_val = y_train_all[val_idx]
        X_test_s = scaler.transform(X_test)

        eeg_dim = X_tr.shape[1]
        encoder = EEGEncoder(eeg_dim, hidden_dim=128).to(device)
        task_clf = TaskClassifier(encoder.output_dim).to(device)

        optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        prototypes = {0: [], 1: []}
        for i in range(len(y_tr)):
            with torch.no_grad():
                z = encoder(X_tr_t[i:i+1])
            prototypes[y_tr[i]].append(z.cpu())

        for cls in prototypes:
            prototypes[cls] = torch.stack(prototypes[cls]).mean(dim=0).to(device)

        best_val_f1 = 0
        best_state = None
        patience = 0

        for epoch in range(50):
            encoder.train()
            task_clf.train()

            z = encoder(X_tr_t)
            task_logits = task_clf(z)

            task_loss = criterion(task_logits, y_tr_t)

            prototypes_matrix = torch.stack([prototypes[0], prototypes[1]])
            cos_sim = torch.mm(z, prototypes_matrix.T)
            labels = torch.LongTensor(y_tr).to(device)
            con_loss = F.cross_entropy(cos_sim / temperature, labels)

            loss = task_loss + lambda_con * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for cls in prototypes:
                mask = (y_tr_t.squeeze() == cls).float().unsqueeze(1)
                if mask.sum() > 0:
                    prototypes[cls] = (z * mask).sum(dim=0) / (mask.sum() + 1e-8)

            encoder.eval()
            task_clf.eval()
            with torch.no_grad():
                val_z = encoder(X_val_t)
                val_logits = task_clf(val_z)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_f1 = f1_score(y_val, val_preds.cpu().numpy(), average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {f'encoder.{k}': v.cpu().clone() for k, v in encoder.state_dict().items()}
                best_state.update({f'task_clf.{k}': v.cpu().clone() for k, v in task_clf.state_dict().items()})
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

        if best_state is not None and len(best_state) > 0:
            encoder.load_state_dict({k.replace('encoder.', ''): v.to(device) for k, v in best_state.items() if k.startswith('encoder.')})
            task_clf.load_state_dict({k.replace('task_clf.', ''): v.to(device) for k, v in best_state.items() if k.startswith('task_clf.')})

        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z = encoder(X_test_t)
            test_logits = task_clf(test_z)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        bacc = balanced_accuracy_score(y_test, test_preds)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', warn_for=[])

        results.append({
            'model': f'EEG_Contrastive_lamb{lambda_con}_temp{temperature}',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'n_train': len(y_tr),
            'n_test': len(y_test)
        })
        print(f"  {held_out}: Acc={acc:.4f}, F1={f1:.4f}")

    return results

def run_gaze_anchor_experiment(seed=0, lambda_anchor=0.1, lambda_kd=0.5):
    print(f"\n--- EEG_GazeAnchor (anch={lambda_anchor}, kd={lambda_kd}) ---")
    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]

        eeg_train, gaze_train, y_train = [], [], []
        for subj in train_subjs:
            eeg_X, gaze_X, eeg_y, gaze_y = load_eeg_gaze_paired(subj)
            if eeg_X is not None:
                common_len = min(len(eeg_X), len(gaze_X))
                eeg_train.append(eeg_X[:common_len])
                gaze_train.append(gaze_X[:common_len])
                y_train.append(eeg_y[:common_len])

        eeg_test, gaze_test, y_test = load_eeg_gaze_paired(held_out)

        if len(eeg_train) == 0 or eeg_test is None or len(eeg_test) == 0:
            continue

        eeg_train = np.vstack(eeg_train)
        gaze_train = np.vstack(gaze_train)
        y_train = np.concatenate(y_train)

        indices = np.random.permutation(len(y_train))
        val_size = int(len(y_train) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()

        eeg_tr = scaler_eeg.fit_transform(eeg_train[train_idx])
        gaze_tr = scaler_gaze.fit_transform(gaze_train[train_idx])
        y_tr = y_train[train_idx]
        eeg_val = scaler_eeg.transform(eeg_train[val_idx])
        gaze_val = scaler_gaze.transform(gaze_train[val_idx])
        y_val = y_train[val_idx]
        eeg_test_s = scaler_eeg.transform(eeg_test)
        gaze_test_s = scaler_gaze.transform(gaze_test)

        eeg_dim = eeg_tr.shape[1]
        gaze_dim = gaze_tr.shape[1]

        gaze_encoder = GazeEncoder(gaze_dim, hidden_dim=64).to(device)
        eeg_encoder = EEGEncoder(eeg_dim, hidden_dim=128).to(device)
        eeg_clf = TaskClassifier(eeg_encoder.output_dim).to(device)
        gaze_clf = TaskClassifier(gaze_encoder.output_dim).to(device)

        optimizer = optim.Adam(list(eeg_encoder.parameters()) + list(eeg_clf.parameters()) + list(gaze_encoder.parameters()) + list(gaze_clf.parameters()), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        eeg_tr_t = torch.FloatTensor(eeg_tr).to(device)
        gaze_tr_t = torch.FloatTensor(gaze_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        eeg_val_t = torch.FloatTensor(eeg_val).to(device)
        gaze_val_t = torch.FloatTensor(gaze_val).to(device)
        eeg_test_t = torch.FloatTensor(eeg_test_s).to(device)
        gaze_test_t = torch.FloatTensor(gaze_test_s).to(device)

        gaze_encoder.eval()
        with torch.no_grad():
            gaze_teacher = gaze_encoder(gaze_tr_t)

        best_val_f1 = 0
        best_state = None
        patience = 0

        for epoch in range(50):
            eeg_encoder.train()
            eeg_clf.train()
            gaze_encoder.train()
            gaze_clf.train()

            z_gaze = gaze_encoder(gaze_tr_t)
            z_eeg = eeg_encoder(eeg_tr_t)
            eeg_logits = eeg_clf(z_eeg)
            gaze_logits = gaze_clf(z_gaze)

            task_loss = criterion(eeg_logits, y_tr_t)
            anchor_loss = F.cosine_embedding_loss(z_eeg, z_gaze.detach(), torch.ones(len(y_tr)).to(device))
            kd_loss = F.kl_div(F.log_softmax(eeg_logits, dim=1), torch.sigmoid(gaze_logits).detach(), reduction='batchmean')

            loss = task_loss + lambda_anchor * anchor_loss + lambda_kd * kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gaze_encoder.eval()
            eeg_encoder.eval()
            eeg_clf.eval()
            with torch.no_grad():
                val_z_eeg = eeg_encoder(eeg_val_t)
                val_logits = eeg_clf(val_z_eeg)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_f1 = f1_score(y_val, val_preds.cpu().numpy(), average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {f'eeg_encoder.{k}': v.cpu().clone() for k, v in eeg_encoder.state_dict().items()}
                best_state.update({f'eeg_clf.{k}': v.cpu().clone() for k, v in eeg_clf.state_dict().items()})
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

        if best_state is not None and len(best_state) > 0:
            eeg_encoder.load_state_dict({k.replace('eeg_encoder.', ''): v.to(device) for k, v in best_state.items() if k.startswith('eeg_encoder.')})
            eeg_clf.load_state_dict({k.replace('eeg_clf.', ''): v.to(device) for k, v in best_state.items() if k.startswith('eeg_clf.')})

        eeg_encoder.eval()
        eeg_clf.eval()
        with torch.no_grad():
            test_z = eeg_encoder(eeg_test_t)
            test_logits = eeg_clf(test_z)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        bacc = balanced_accuracy_score(y_test, test_preds)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', warn_for=[])

        results.append({
            'model': f'EEG_GazeAnchor_a{lambda_anchor}_kd{lambda_kd}',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'n_train': len(y_tr),
            'n_test': len(y_test)
        })
        print(f"  {held_out}: Acc={acc:.4f}, F1={f1:.4f}")

    return results

def main():
    print("="*70)
    print("EEG Subject-Adaptation Pilot (seed=0)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    seed = 0
    all_results = []

    all_results.extend(run_coral_experiment(seed=seed))

    for lambda_adv in [0.01, 0.05, 0.1]:
        all_results.extend(run_adversarial_experiment(seed=seed, lambda_adv=lambda_adv))

    for lambda_con in [0.01, 0.05, 0.1]:
        all_results.extend(run_contrastive_experiment(seed=seed, lambda_con=lambda_con, temperature=0.07))

    for lambda_anchor in [0.01, 0.05, 0.1]:
        for lambda_kd in [0.1, 0.5, 1.0]:
            all_results.extend(run_gaze_anchor_experiment(seed=seed, lambda_anchor=lambda_anchor, lambda_kd=lambda_kd))

    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(RESULTS_DIR, f"eeg_adaptation_pilot_seed{seed}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    summary.columns = ['model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'bacc_mean', 'bacc_std']
    summary = summary.sort_values('accuracy_mean', ascending=False)

    summary_csv = os.path.join(RESULTS_DIR, f"eeg_adaptation_summary_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)

    print("\n" + "="*70)
    print("EEG ADAPTATION SUMMARY")
    print("="*70)
    print(f"{'Model':<40} {'Acc Mean':>10} {'Acc Std':>10} {'F1 Mean':>10}")
    print("-"*70)
    for _, row in summary.iterrows():
        print(f"{row['model']:<40} {row['accuracy_mean']:>10.4f} {row['accuracy_std']:>10.4f} {row['macro_f1_mean']:>10.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()