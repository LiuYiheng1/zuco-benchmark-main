"""
EEG Subject Leakage / Subject-Invariance Analysis (Simplified - 1 seed)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

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

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

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

print("="*70)
print("EEG Subject Leakage / Subject-Invariance Analysis")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

all_results = []
seed = 0

print(f"\n--- Seed {seed} ---")

for held_out in Y_SUBJECTS:
    print(f"\n{held_out}:", flush=True)

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
    X_test_s = scaler.transform(X_test)

    print(f"  Training raw EEG...", flush=True)
    raw_clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
    raw_clf.fit(X_tr, y_train_all[train_idx])
    raw_task_pred = raw_clf.predict(X_test_s)
    raw_task_acc = accuracy_score(y_test, raw_task_pred)
    raw_task_f1 = f1_score(y_test, raw_task_pred, average='macro')

    sub_clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
    sub_clf.fit(X_tr, sub_ids[train_idx])
    raw_sub_pred = sub_clf.predict(X_test_s)
    raw_sub_acc = accuracy_score(np.zeros(len(y_test)), raw_sub_pred)

    print(f"  Training adversarial λ=0.01...", flush=True)
    n_subjects = len(Y_SUBJECTS) - 1
    eeg_dim = X_tr.shape[1]

    encoder = EEGEncoder(eeg_dim).to(device)
    task_clf = TaskClassifier(encoder.output_dim).to(device)
    sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_train_all[train_idx]).unsqueeze(1).to(device)
    sub_tr_t = torch.LongTensor(sub_ids[train_idx]).to(device)

    lambda_adv = 0.01
    for epoch in range(50):
        encoder.train()
        task_clf.train()
        sub_disc.train()

        z = encoder(X_tr_t)
        task_logits = task_clf(z)
        reversed_z = GradientReversalFunction.apply(z, lambda_adv)
        sub_logits = sub_disc(reversed_z)

        task_loss = criterion(task_logits, y_tr_t)
        sub_loss = nn.CrossEntropyLoss()(sub_logits, sub_tr_t)
        loss = task_loss + lambda_adv * sub_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    task_clf.eval()
    with torch.no_grad():
        test_emb = encoder(torch.FloatTensor(X_test_s).to(device)).detach().cpu().numpy()

    adv_task_clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
    train_emb = encoder(torch.FloatTensor(X_tr).to(device)).detach().cpu().numpy()
    adv_task_clf.fit(train_emb, y_train_all[train_idx])
    adv_task_pred = adv_task_clf.predict(test_emb)
    adv_task_acc = accuracy_score(y_test, adv_task_pred)
    adv_task_f1 = f1_score(y_test, adv_task_pred, average='macro')

    adv_sub_clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
    adv_sub_clf.fit(train_emb, sub_ids[train_idx])
    adv_sub_pred = adv_sub_clf.predict(test_emb)
    adv_sub_acc = accuracy_score(sub_ids[train_idx][:len(adv_sub_pred)], adv_sub_pred)

    print(f"  Raw: Sub={raw_sub_acc:.3f} Task={raw_task_acc:.3f} | Adv01: Sub={adv_sub_acc:.3f} Task={adv_task_acc:.3f}")

    all_results.append({
        'seed': seed,
        'held_out': held_out,
        'Raw_EEG_subject_acc': raw_sub_acc,
        'Raw_EEG_task_acc': raw_task_acc,
        'Raw_EEG_task_f1': raw_task_f1,
        'Adv01_subject_acc': adv_sub_acc,
        'Adv01_task_acc': adv_task_acc,
        'Adv01_task_f1': adv_task_f1,
    })

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "subject_leakage_analysis.csv"), index=False)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary = {
    'Model': ['Raw_EEG', 'EEG_Adversarial λ=0.01'],
    'Subject_Acc_Mean': [results_df['Raw_EEG_subject_acc'].mean(), results_df['Adv01_subject_acc'].mean()],
    'Task_Acc_Mean': [results_df['Raw_EEG_task_acc'].mean(), results_df['Adv01_task_acc'].mean()],
    'Task_F1_Mean': [results_df['Raw_EEG_task_f1'].mean(), results_df['Adv01_task_f1'].mean()],
}
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

raw_sub = results_df['Raw_EEG_subject_acc'].mean()
adv01_sub = results_df['Adv01_subject_acc'].mean()
raw_task = results_df['Raw_EEG_task_acc'].mean()
adv01_task = results_df['Adv01_task_acc'].mean()

print(f"\nSubject Identity Predictability (lower = more subject-invariant):")
print(f"  Raw EEG: {raw_sub:.4f}")
print(f"  Adversarial λ=0.01: {adv01_sub:.4f} (Δ={adv01_sub-raw_sub:.4f})")

print(f"\nTask Classification (higher = better):")
print(f"  Raw EEG: {raw_task:.4f}")
print(f"  Adversarial λ=0.01: {adv01_task:.4f} (Δ={adv01_task-raw_task:.4f})")

if adv01_sub < raw_sub:
    print("\n✓ Adversarial training reduces subject identity predictability!")
else:
    print("\n✗ Adversarial training does NOT reduce subject identity predictability")

if adv01_task >= raw_task:
    print("✓ Adversarial training maintains or improves task classification!")
else:
    print("✗ Adversarial training hurts task classification")

print(f"\nSaved to {RESULTS_DIR}/subject_leakage_analysis.csv")