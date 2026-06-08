"""
EEG Subject Leakage / Subject-Invariance Analysis (FIXED)
Correct protocol: Within-subject CV on training subjects only
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

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

def train_adversarial_encoder(X_train, y_train, sub_ids, seed, lambda_adv, device, n_subjects):
    """Train adversarial encoder on training data"""
    np.random.seed(seed)
    indices = np.random.permutation(len(y_train))
    val_size = int(len(y_train) * 0.1)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[train_idx])
    y_tr = y_train[train_idx]
    sub_tr = sub_ids[train_idx]
    X_val = scaler.transform(X_train[val_idx])
    y_val = y_train[val_idx]

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
        sub_loss = nn.CrossEntropyLoss()(sub_logits, sub_tr_t)
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
    with torch.no_grad():
        train_emb = encoder(torch.FloatTensor(X_tr).to(device)).cpu().numpy()
        val_emb = encoder(X_val_t).cpu().numpy()

    return encoder, scaler, train_emb, val_emb, train_idx, val_idx

def evaluate_subject_classification_cv(embeddings, sub_ids, n_folds=5):
    """Evaluate subject classification using k-fold CV on training subjects"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_true = []
    all_pred = []

    for train_idx, val_idx in kf.split(embeddings):
        clf = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)
        clf.fit(embeddings[train_idx], sub_ids[train_idx])
        pred = clf.predict(embeddings[val_idx])
        all_true.extend(sub_ids[val_idx])
        all_pred.extend(pred)

    acc = accuracy_score(all_true, all_pred)
    bacc = balanced_accuracy_score(all_true, all_pred)
    return acc, bacc

def main():
    print("="*70)
    print("EEG Subject Leakage / Subject-Invariance Analysis (FIXED)")
    print("="*70)
    print("\nCorrected Protocol:")
    print("- Within-subject CV on training subjects only")
    print("- Compare subject predictability of raw EEG vs adversarial embeddings")
    print("- Use balanced accuracy for multi-class subject classification")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    all_results = []
    seeds = [0, 1, 2]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        held_out = Y_SUBJECTS[seed % len(Y_SUBJECTS)]
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
        n_subjects = len(train_subjs)

        print(f"  Train subjects: {n_subjects}, Test subject: {held_out}")
        print(f"  Train samples: {len(y_train_all)}, Test samples: {len(y_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_all)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n  [1] Raw EEG features - subject classification CV...")
        raw_sub_acc, raw_sub_bacc = evaluate_subject_classification_cv(X_train_scaled, sub_ids)
        print(f"      Raw EEG Subject Acc: {raw_sub_acc:.4f}, Balanced Acc: {raw_sub_bacc:.4f}")

        print(f"  [2] Adversarial embeddings - subject classification CV...")
        encoder, encoder_scaler, train_emb, val_emb, train_idx, val_idx = train_adversarial_encoder(
            X_train_all, y_train_all, sub_ids, seed, 0.01, device, n_subjects
        )
        adv_sub_acc, adv_sub_bacc = evaluate_subject_classification_cv(train_emb, sub_ids)
        print(f"      Adversarial Subject Acc: {adv_sub_acc:.4f}, Balanced Acc: {adv_sub_bacc:.4f}")

        print(f"\n  Subject predictability comparison:")
        print(f"    Raw EEG: {raw_sub_bacc:.4f} (balanced acc)")
        print(f"    Adversarial: {adv_sub_bacc:.4f} (balanced acc)")
        print(f"    Δ = {adv_sub_bacc - raw_sub_bacc:+.4f}")

        all_results.append({
            'seed': seed,
            'test_subject': held_out,
            'n_train_subjects': n_subjects,
            'raw_subject_acc': raw_sub_acc,
            'raw_subject_bacc': raw_sub_bacc,
            'adv_subject_acc': adv_sub_acc,
            'adv_subject_bacc': adv_sub_bacc,
        })

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "subject_leakage_analysis_fixed.csv")
    results_df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY (across seeds)")
    print("="*70)
    print(f"\nRaw EEG - Subject Classification:")
    print(f"  Mean Acc: {results_df['raw_subject_acc'].mean():.4f}")
    print(f"  Mean Balanced Acc: {results_df['raw_subject_bacc'].mean():.4f}")

    print(f"\nAdversarial - Subject Classification:")
    print(f"  Mean Acc: {results_df['adv_subject_acc'].mean():.4f}")
    print(f"  Mean Balanced Acc: {results_df['adv_subject_bacc'].mean():.4f}")

    raw_mean = results_df['raw_subject_bacc'].mean()
    adv_mean = results_df['adv_subject_bacc'].mean()

    print(f"\nDifference (Adversarial - Raw):")
    print(f"  Δ = {adv_mean - raw_mean:+.4f}")

    if adv_mean < raw_mean:
        print(f"\n✓ Adversarial embeddings are LESS predictable for subject identity")
        print(f"  This SUPPORTS the claim that adversarial training reduces subject-specific information")
    elif adv_mean > raw_mean:
        print(f"\n✗ Adversarial embeddings are MORE predictable for subject identity")
        print(f"  This CONTRADICTS the claim that adversarial training reduces subject-specific information")
    else:
        print(f"\n? No difference in subject predictability")

    print(f"\nSaved to: {output_path}")

if __name__ == '__main__':
    main()