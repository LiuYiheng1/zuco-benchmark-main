"""
CAET: Calibration-Augmented EEG Training

This module implements data augmentation techniques to improve low-shot EEG calibration.

Augmentations:
1. Gaussian noise
2. Feature dropout
3. Same-class mixup
4. Combined (noise + dropout + mixup)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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

class EEGClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def gaussian_noise_augment(X, noise_std=0.1):
    """Add Gaussian noise to features"""
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def feature_dropout_augment(X, dropout_rate=0.2):
    """Randomly dropout features"""
    mask = np.random.binomial(1, 1 - dropout_rate, X.shape)
    return X * mask

def mixup_augment(X, y, alpha=0.3):
    """Same-class mixup augmentation"""
    X_aug = []
    y_aug = []

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        n = len(idx)

        if n < 2:
            X_aug.extend(X[idx])
            y_aug.extend(y[idx])
            continue

        for i in range(n):
            j = np.random.choice([k for k in range(n) if k != i])
            lam = np.random.beta(alpha, alpha)
            x_mix = lam * X[idx[i]] + (1 - lam) * X[idx[j]]
            X_aug.append(x_mix)
            y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

def train_and_evaluate(X_cal, y_cal, X_test, y_test, model_type, device='cpu'):
    """Train model with specified augmentation and evaluate"""

    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    if model_type == 'EEG_MLP':
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

    elif model_type in ['EEG_MLP_CAET_noise', 'EEG_MLP_CAET_dropout', 'EEG_MLP_CAET_mixup', 'EEG_MLP_CAET_combo']:
        input_dim = X_cal_s.shape[1]
        model = EEGClassifier(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_cal_t = torch.FloatTensor(X_cal_s).to(device)
        y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        n_aug = 50

        if model_type == 'EEG_MLP_CAET_noise':
            for epoch in range(100):
                model.train()
                X_aug = gaussian_noise_augment(X_cal_s, noise_std=0.1)
                X_aug_t = torch.FloatTensor(X_aug).to(device)
                y_aug_t = y_cal_t

                logits = model(X_aug_t)
                loss = criterion(logits, y_aug_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif model_type == 'EEG_MLP_CAET_dropout':
            for epoch in range(100):
                model.train()
                X_aug = feature_dropout_augment(X_cal_s, dropout_rate=0.2)
                X_aug_t = torch.FloatTensor(X_aug).to(device)
                y_aug_t = y_cal_t

                logits = model(X_aug_t)
                loss = criterion(logits, y_aug_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif model_type == 'EEG_MLP_CAET_mixup':
            for epoch in range(100):
                model.train()
                X_mix, y_mix = mixup_augment(X_cal_s, y_cal, alpha=0.3)
                X_mix_t = torch.FloatTensor(X_mix).to(device)
                y_mix_t = torch.FloatTensor(y_mix).unsqueeze(1).to(device)

                logits = model(X_mix_t)
                loss = criterion(logits, y_mix_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif model_type == 'EEG_MLP_CAET_combo':
            for epoch in range(100):
                model.train()

                X_noise = gaussian_noise_augment(X_cal_s, noise_std=0.1)
                X_drop = feature_dropout_augment(X_cal_s, dropout_rate=0.2)
                X_mix, y_mix = mixup_augment(X_cal_s, y_cal, alpha=0.3)

                X_combined = np.vstack([X_noise, X_drop, X_mix])
                y_combined = np.concatenate([y_cal, y_cal, y_mix])

                indices = np.random.permutation(len(y_combined))[:len(y_cal) * 2]
                X_combined = X_combined[indices]
                y_combined = y_combined[indices]

                X_combined_t = torch.FloatTensor(X_combined).to(device)
                y_combined_t = torch.FloatTensor(y_combined).unsqueeze(1).to(device)

                logits = model(X_combined_t)
                loss = criterion(logits, y_combined_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        bacc = balanced_accuracy_score(y_test, preds)
        try:
            auroc = roc_auc_score(y_test, probs)
        except:
            auroc = 0.5

        return acc, f1, bacc, auroc

def run_experiment(seed, model_type):
    """Run CAET experiment"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for held_out in Y_SUBJECTS:
        print(f"\n    {model_type} - {held_out}:", flush=True)

        X_eeg, y_eeg = load_eeg_data(held_out)
        if X_eeg is None or len(X_eeg) < 50:
            continue

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

            acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test, model_type, device)

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

            print(f"      {n_cal_per_class}-shot: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}", flush=True)

    return results

def main():
    print("="*70)
    print("CAET: Calibration-Augmented EEG Training Experiment")
    print("="*70)

    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []
    model_types = ['EEG_MLP', 'EEG_MLP_CAET_noise', 'EEG_MLP_CAET_dropout', 'EEG_MLP_CAET_mixup', 'EEG_MLP_CAET_combo']

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            results = run_experiment(seed, model_type)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "caet_augmentation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "caet_augmentation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()