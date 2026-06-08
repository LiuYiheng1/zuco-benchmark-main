"""
EEG Subject-Adaptation Pilot - Simplified version
Runs only CORAL and Adversarial (which worked), skips Contrastive and GazeAnchor
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
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

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return x

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

def coral_align(X_source, X_target):
    cov_src = np.cov(X_source.T)
    cov_tgt = np.cov(X_target.T)

    eigval_src, eigvec_src = np.linalg.eigh(cov_src)
    order_src = eigval_src.argsort()[::-1]
    eigval_src = eigval_src[order_src]
    eigvec_src = eigvec_src[:, order_src]

    d_src = np.diag(1.0 / np.sqrt(np.maximum(eigval_src, 1e-8)))
    whitening_src = eigvec_src @ d_src @ eigvec_src.T

    eigval_tgt, eigvec_tgt = np.linalg.eigh(cov_tgt)
    order_tgt = eigval_tgt.argsort()[::-1]
    eigval_tgt = eigval_tgt[order_tgt]
    eigvec_tgt = eigvec_tgt[:, order_tgt]

    d_tgt = np.diag(1.0 / np.sqrt(np.maximum(eigval_tgt, 1e-8)))
    whitening_tgt = eigvec_tgt @ d_tgt @ eigvec_tgt.T

    X_src_aligned = X_source @ whitening_src
    X_tgt_aligned = X_target @ whitening_src

    return X_src_aligned, X_tgt_aligned

def run_coral_experiment(seed=0):
    print("\n--- EEG_CORAL ---")
    results = []
    np.random.seed(seed)

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_eeg_data(held_out)

        if len(X_train_all) == 0 or X_test is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_all[train_idx])
        X_val_s = scaler.transform(X_train_all[val_idx])
        X_test_s = scaler.transform(X_test)

        X_tr_aligned, X_test_aligned = coral_align(X_tr_s, X_test_s)

        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_tr_aligned, y_train_all[train_idx])
        y_pred = clf.predict(X_test_aligned)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append({
            'model': 'EEG_CORAL',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'n_test': len(y_test)
        })
        print(f"  {held_out}: Acc={acc:.4f}")

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

        if len(X_train_all) == 0 or X_test is None:
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
        sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

        optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
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

        for epoch in range(50):
            encoder.train()
            task_clf.train()
            sub_disc.train()

            z = encoder(X_tr_t)
            task_logits = task_clf(z)
            reversed_z = GradientReversalFunction.apply(z, lambda_adv)
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
                best_encoder_state = encoder.state_dict().copy()
                best_clf_state = task_clf.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state)
            task_clf.load_state_dict(best_clf_state)

        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z = encoder(X_test_t)
            test_logits = task_clf(test_z)
            test_preds = (torch.sigmoid(test_logits) > 0.5).float()

        acc = accuracy_score(y_test, test_preds.cpu().numpy())
        f1 = f1_score(y_test, test_preds.cpu().numpy(), average='macro')

        results.append({
            'model': f'EEG_Adversarial_lamb{lambda_adv}',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'n_test': len(y_test)
        })
        print(f"  {held_out}: Acc={acc:.4f}")

    return results

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def main():
    print("="*60)
    print("EEG Subject-Adaptation Pilot (CORAL + Adversarial)")
    print("="*60)

    seed = 0
    all_results = []

    all_results.extend(run_coral_experiment(seed=seed))
    for lambda_adv in [0.01, 0.05, 0.1]:
        all_results.extend(run_adversarial_experiment(seed=seed, lambda_adv=lambda_adv))

    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(RESULTS_DIR, f"eeg_adaptation_pilot_seed{seed}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std']
    }).reset_index()
    summary.columns = ['model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std']
    summary = summary.sort_values('accuracy_mean', ascending=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for _, row in summary.iterrows():
        print(f"{row['model']:25s}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()