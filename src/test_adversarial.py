import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = "features"
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

print("Loading data...")
train_subjs = [s for s in Y_SUBJECTS if s != 'YAC']
X_train_all, y_train_all, sub_ids = [], [], []

for subj_idx, subj in enumerate(train_subjs):
    X, y = load_eeg_data(subj)
    if X is not None:
        X_train_all.append(X)
        y_train_all.append(y)
        sub_ids.extend([subj_idx] * len(y))

X_test, y_test = load_eeg_data('YAC')
print(f"Train: {np.vstack(X_train_all).shape}, Test: {X_test.shape}")

X_train_all = np.vstack(X_train_all)
y_train_all = np.concatenate(y_train_all)
sub_ids = np.array(sub_ids)

np.random.seed(0)
indices = np.random.permutation(len(y_train_all))
val_size = int(len(y_train_all) * 0.1)
train_idx = indices[val_size:]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train_all[train_idx])
y_tr = y_train_all[train_idx]
sub_tr = sub_ids[train_idx]
X_test_s = scaler.transform(X_test)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

eeg_dim = X_tr.shape[1]
n_subjects = len(Y_SUBJECTS) - 1
lambda_adv = 0.01

encoder = EEGEncoder(eeg_dim).to(device)
task_clf = TaskClassifier(encoder.output_dim).to(device)
sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

X_tr_t = torch.FloatTensor(X_tr).to(device)
y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
sub_tr_t = torch.LongTensor(sub_tr).to(device)
X_test_t = torch.FloatTensor(X_test_s).to(device)

print("Training adversarial model...")
for epoch in range(10):
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

    if epoch % 5 == 0:
        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z = encoder(X_test_t)
            test_logits = task_clf(test_z)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)
        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, TaskLoss={task_loss.item():.4f}, SubLoss={sub_loss.item():.4f}, Test Acc={acc:.4f}, F1={f1:.4f}")

print("Done!")