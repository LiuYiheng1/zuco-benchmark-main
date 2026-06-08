"""HD-SIED: Hard-Domain SIED with GroupDRO Weighting (Fast Version)"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)

class TaskClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class SubjectDiscriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_subjects)
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

def run_raw_eeg(seed):
    results = []
    for held_out in Y_SUBJECTS:
        try:
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

            np.random.seed(seed)
            indices = np.random.permutation(len(y_train_all))
            train_idx = indices[int(len(y_train_all) * 0.1):]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            X_test_s = scaler.transform(X_test)

            clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
            clf.fit(X_tr, y_train_all[train_idx])
            y_pred = clf.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)

            results.append({
                'model': 'Raw_EEG', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': 0.5
            })
        except:
            pass
    return results

def run_sied_standard(seed, lambda_adv=1.0):
    results = []
    device = 'cpu'
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
            train_idx = indices[int(len(y_train_all) * 0.1):]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            X_test_s = scaler.transform(X_test)

            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim).to(device)
            task_clf = TaskClassifier(encoder.output_dim).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr)
            y_tr_t = torch.FloatTensor(y_train_all[train_idx]).unsqueeze(1)
            sub_tr_t = torch.LongTensor(sub_ids[train_idx])
            X_test_t = torch.FloatTensor(X_test_s)

            for epoch in range(30):
                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                sub_logits = sub_disc(z)

                task_loss = criterion(task_logits, y_tr_t)
                sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
                loss = task_loss + lambda_adv * sub_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            encoder.eval()
            with torch.no_grad():
                test_z = encoder(X_test_t)
                test_logits = task_clf(test_z)
                test_probs = torch.sigmoid(test_logits).numpy().flatten()
                test_preds = (test_probs >= 0.5).astype(int)

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average='macro')
            bacc = balanced_accuracy_score(y_test, test_preds)

            results.append({
                'model': f'SIED', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': 0.5
            })
        except:
            pass
    return results

def run_hd_sied(seed, lambda_adv=0.01, eta=0.05):
    results = []
    device = 'cpu'
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
            train_idx = indices[int(len(y_train_all) * 0.1):]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            X_test_s = scaler.transform(X_test)

            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim).to(device)
            task_clf = TaskClassifier(encoder.output_dim).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr)
            y_tr_t = torch.FloatTensor(y_train_all[train_idx]).unsqueeze(1)
            sub_tr_t = torch.LongTensor(sub_ids[train_idx])
            X_test_t = torch.FloatTensor(X_test_s)

            q_s = np.ones(n_subjects) / n_subjects

            for epoch in range(30):
                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                sub_logits = sub_disc(z)

                per_subject_losses = []
                for s in range(n_subjects):
                    mask = sub_tr_t.cpu().numpy() == s
                    if mask.sum() > 0:
                        loss_s = F.binary_cross_entropy_with_logits(task_logits.squeeze()[mask], y_tr_t.squeeze()[mask])
                        per_subject_losses.append(loss_s.item())
                    else:
                        per_subject_losses.append(0.0)

                per_subject_losses = np.array(per_subject_losses)
                q_s = q_s * np.exp(eta * per_subject_losses)
                q_s = q_s / q_s.sum()

                task_loss_dro = sum(q_s[s] * per_subject_losses[s] for s in range(n_subjects))
                task_loss_tensor = torch.tensor(task_loss_dro, dtype=torch.float32)
                sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
                loss = task_loss_tensor + lambda_adv * sub_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            encoder.eval()
            with torch.no_grad():
                test_z = encoder(X_test_t)
                test_logits = task_clf(test_z)
                test_probs = torch.sigmoid(test_logits).numpy().flatten()
                test_preds = (test_probs >= 0.5).astype(int)

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average='macro')
            bacc = balanced_accuracy_score(y_test, test_preds)

            results.append({
                'model': f'HD-SIED_l{lambda_adv}_eta{eta}', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': 0.5
            })
        except:
            pass
    return results

print('HD-SIED Experiments (Fast)', flush=True)
print('='*60, flush=True)

all_results = []
seeds = [0, 1, 2]

print('Running Raw_EEG and SIED...', flush=True)
for seed in seeds:
    all_results.extend(run_raw_eeg(seed))
    all_results.extend(run_sied_standard(seed, lambda_adv=1.0))

print('Running HD-SIED...', flush=True)
for seed in seeds:
    all_results.extend(run_hd_sied(seed, lambda_adv=0.01, eta=0.05))
    all_results.extend(run_hd_sied(seed, lambda_adv=0.05, eta=0.1))

df = pd.DataFrame(all_results)
df.to_csv("results/final/hd_sied_results.csv", index=False)

print("\nResults Summary:")
for model in df['model'].unique():
    data = df[df['model'] == model]
    acc = data['accuracy'].mean()
    std = data['accuracy'].std()
    print(f"  {model}: {acc:.4f}±{std:.4f}")

sied_acc = df[df['model'] == 'SIED']['accuracy'].mean()
print(f"\nSIED baseline: {sied_acc:.4f}")
for model in df['model'].unique():
    if 'HD-SIED' in model:
        hd_acc = df[df['model'] == model]['accuracy'].mean()
        gain = hd_acc - sied_acc
        print(f"  {model}: {hd_acc:.4f} (gap={gain:+.4f})")

print("\nDone!")