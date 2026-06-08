"""SIED Stability - Simplified Version"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"

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

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)

class TaskClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class SubjectDiscriminator(nn.Module):
    def __init__(self, input_dim, n_subjects, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            GradientReversalLayer(1.0),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_subjects)
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

def compute_lambda_warmup(p, lambda_max, gamma):
    return lambda_max * (2 / (1 + np.exp(-gamma * p)) - 1)

def run_sied(seed, lambda_adv, dropout, use_warmup=False, lambda_max=0.01, gamma=5):
    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    for held_out in Y_SUBJECTS:
        try:
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

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            y_tr = y_train_all[train_idx]
            sub_tr = sub_ids[train_idx]
            X_test_s = scaler.transform(X_test)

            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim, dropout=dropout).to(device)
            task_clf = TaskClassifier(encoder.output_dim, dropout=dropout).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects, dropout=dropout).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr).to(device)
            y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
            sub_tr_t = torch.LongTensor(sub_tr).to(device)
            X_test_t = torch.FloatTensor(X_test_s).to(device)

            n_epochs = 50
            for epoch in range(n_epochs):
                if use_warmup:
                    p = epoch / n_epochs
                    current_lambda = compute_lambda_warmup(p, lambda_max, gamma)
                else:
                    current_lambda = lambda_adv

                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                sub_logits = sub_disc(z)

                task_loss = criterion(task_logits, y_tr_t)
                sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
                loss = task_loss + current_lambda * sub_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            encoder.eval()
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

            test_z_np = test_z.cpu().numpy()
            try:
                sub_clf = LogisticRegression(max_iter=1000, random_state=seed)
                sub_clf.fit(test_z_np, y_test)
                sub_pred = sub_clf.predict(test_z_np)
                sub_acc = accuracy_score(y_test, sub_pred)
            except:
                sub_acc = 0.5

            model_name = f'SIED_warmup_lmax{lambda_max}_g{gamma}_d{dropout}' if use_warmup else f'SIED_fixed_l{lambda_adv}_d{dropout}'

            results.append({
                'model': model_name,
                'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'subject_predictability': sub_acc,
                'use_warmup': use_warmup, 'lambda_adv': lambda_adv, 'dropout': dropout
            })

        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

print("SIED Stability Optimization", flush=True)
print("="*60, flush=True)

all_results = []
seeds = [0, 1, 2]

print("\nBaseline experiments:", flush=True)
for seed in seeds:
    print(f"  Seed {seed}", flush=True)
    all_results.extend(run_sied(seed, lambda_adv=0.01, dropout=0.3, use_warmup=False))

print("\nWarmup experiments:", flush=True)
for seed in seeds:
    print(f"  Seed {seed}", flush=True)
    all_results.extend(run_sied(seed, lambda_adv=0.01, dropout=0.3, use_warmup=True, lambda_max=0.01, gamma=5))
    all_results.extend(run_sied(seed, lambda_adv=0.01, dropout=0.3, use_warmup=True, lambda_max=0.05, gamma=10))

df = pd.DataFrame(all_results)
df.to_csv(f"{RESULTS_DIR}/sied_stability_results.csv", index=False)

print("\n" + "="*60, flush=True)
print("Results Summary", flush=True)
print("="*60, flush=True)

for model in df['model'].unique():
    data = df[df['model'] == model]
    if len(data) > 0:
        acc = data['accuracy'].mean()
        std = data['accuracy'].std()
        f1 = data['macro_f1'].mean()
        bacc = data['balanced_accuracy'].mean()
        sub_pred = data['subject_predictability'].mean()
        print(f"{model}: acc={acc:.4f}±{std:.4f}, f1={f1:.4f}, bacc={bacc:.4f}, sub_pred={sub_pred:.4f}", flush=True)

print("\nDone!", flush=True)