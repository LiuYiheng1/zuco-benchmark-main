"""
PyTorch LOSO-Y Baseline Matrix
EEG MLP, Gaze MLP, Early Concat, Late Fusion, Attention Fusion
Aligned with SVM baseline settings
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_features(subject, feature_name):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split("_")
    if len(parts) >= 2 and parts[1] == "NR":
        return "NR", True
    elif len(parts) >= 2 and parts[1] == "TSR":
        return "TSR", True
    return "", False

def load_labeled_data(subjects, feature_name):
    all_X, all_y = [], []
    for subj in subjects:
        feats = load_features(subj, feature_name)
        if feats is None:
            continue
        for key, values in feats.items():
            label, is_labeled = parse_key(key)
            if not is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
    return np.array(all_X), np.array(all_y)

def load_combined_eeg_gaze(subjects):
    all_X, all_y = [], []
    for subj in subjects:
        feats = load_features(subj, 'sent_gaze_sacc_eeg_means')
        if feats is None:
            continue
        for key, values in feats.items():
            label, is_labeled = parse_key(key)
            if not is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
    return np.array(all_X), np.array(all_y)

def get_feature_dimensions():
    combined_sample = load_features('YAC', 'sent_gaze_sacc_eeg_means')
    combined_dim = len(list(combined_sample.values())[0]) if combined_sample else 0
    eeg_sample = load_features('YAC', 'electrode_features_all')
    eeg_dim = len(list(eeg_sample.values())[0][:-1]) if eeg_sample else 0
    gaze_sample = load_features('YAC', 'sent_gaze_sacc')
    gaze_dim = len(list(gaze_sample.values())[0][:-1]) if gaze_sample else 0
    return combined_dim, eeg_dim, gaze_dim

class SimpleFusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

class DualBranchNet(nn.Module):
    def __init__(self, eeg_dim, gaze_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.eeg_net = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gaze_net = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, eeg, gaze):
        eeg_h = self.eeg_net(eeg)
        gaze_h = self.gaze_net(gaze)
        fused = torch.cat([eeg_h, gaze_h], dim=1)
        return self.fusion(fused)

class AttentionFusionNet(nn.Module):
    def __init__(self, eeg_dim, gaze_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.eeg_net = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gaze_net = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, eeg, gaze):
        eeg_h = self.eeg_net(eeg)
        gaze_h = self.gaze_net(gaze)
        concat = torch.cat([eeg_h, gaze_h], dim=1)
        attn = self.attention(concat)
        weighted = concat * attn
        return self.fusion(weighted)

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def train_dual_model(model, eeg_train, gaze_train, y_train, eeg_val, gaze_val, y_val, epochs=50, batch_size=128, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    eeg_train_t = torch.FloatTensor(eeg_train).to(device)
    gaze_train_t = torch.FloatTensor(gaze_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    eeg_val_t = torch.FloatTensor(eeg_val).to(device)
    gaze_val_t = torch.FloatTensor(gaze_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_dataset = TensorDataset(eeg_train_t, gaze_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_eeg, batch_gaze, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_eeg, batch_gaze)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(eeg_val_t, gaze_val_t)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def evaluate_model(model, X_test, y_test, device='cpu'):
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_t)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, preds, average='macro', warn_for=[])
    cm = confusion_matrix(y_test, preds)

    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return {
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'precision_macro': prec,
        'recall_macro': rec,
        'auroc': auroc,
        'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
        'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
        'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
        'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0
    }

def evaluate_dual_model(model, eeg_test, gaze_test, y_test, device='cpu'):
    model.eval()
    eeg_t = torch.FloatTensor(eeg_test).to(device)
    gaze_t = torch.FloatTensor(gaze_test).to(device)
    with torch.no_grad():
        outputs = model(eeg_t, gaze_t)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, preds, average='macro', warn_for=[])
    cm = confusion_matrix(y_test, preds)

    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return {
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'precision_macro': prec,
        'recall_macro': rec,
        'auroc': auroc,
        'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
        'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
        'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
        'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0
    }

def run_single_modality(model_name, feature_name, seed=1, device='cpu'):
    sys.stdout.write(f"  {model_name} ({feature_name}, seed={seed})...\n")
    sys.stdout.flush()
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]

        X_train, y_train = load_labeled_data(train_subjs, feature_name)
        X_test, y_test = load_labeled_data([held_out], feature_name)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        indices = np.arange(len(y_train))
        indices = shuffle(indices, random_state=seed)
        val_size = int(len(y_train) * 0.1)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        input_dim = X_tr_s.shape[1]
        model = SimpleFusionNet(input_dim)
        model = train_model(model, X_tr_s, y_tr, X_val_s, y_val, epochs=50, device=device)
        metrics = evaluate_model(model, X_test_s, y_test, device=device)

        metrics['model'] = model_name
        metrics['seed'] = seed
        metrics['held_out'] = held_out
        metrics['n_train'] = len(y_tr)
        metrics['n_val'] = len(y_val)
        metrics['n_test'] = len(y_test)
        results.append(metrics)

        sys.stdout.write(f"    {held_out}: Acc={metrics['accuracy']:.4f}\n")
        sys.stdout.flush()

    return results

def run_combined_fusion(model_name, seed=1, device='cpu'):
    sys.stdout.write(f"  {model_name} (Combined, seed={seed})...\n")
    sys.stdout.flush()
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]

        X_train, y_train = load_combined_eeg_gaze(train_subjs)
        X_test, y_test = load_combined_eeg_gaze([held_out])

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        indices = np.arange(len(y_train))
        indices = shuffle(indices, random_state=seed)
        val_size = int(len(y_train) * 0.1)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        input_dim = X_tr_s.shape[1]
        model = SimpleFusionNet(input_dim)
        model = train_model(model, X_tr_s, y_tr, X_val_s, y_val, epochs=50, device=device)
        metrics = evaluate_model(model, X_test_s, y_test, device=device)

        metrics['model'] = model_name
        metrics['seed'] = seed
        metrics['held_out'] = held_out
        metrics['n_train'] = len(y_tr)
        metrics['n_val'] = len(y_val)
        metrics['n_test'] = len(y_test)
        results.append(metrics)

        sys.stdout.write(f"    {held_out}: Acc={metrics['accuracy']:.4f}\n")
        sys.stdout.flush()

    return results

def main():
    print("="*70)
    print("PyTorch LOSO-Y Baseline Matrix")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    combined_dim, eeg_dim, gaze_dim = get_feature_dimensions()
    print(f"Feature dimensions: combined={combined_dim}, eeg={eeg_dim}, gaze={gaze_dim}")

    seeds = [0, 1, 2, 3, 4]
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        all_results.extend(run_single_modality('MLP_EEG_only', 'electrode_features_all', seed, device))
        all_results.extend(run_single_modality('MLP_Gaze_only', 'sent_gaze_sacc', seed, device))
        all_results.extend(run_combined_fusion('MLP_EarlyConcat', seed, device))
        all_results.extend(run_combined_fusion('MLP_LateFusion', seed, device))
        all_results.extend(run_combined_fusion('MLP_AttentionFusion', seed, device))

    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_csv = os.path.join(RESULTS_DIR, f"pytorch_baselines_loso_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    summary_csv = os.path.join(RESULTS_DIR, f"pytorch_summary_mean_std_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    print("\n" + "="*70)
    print("SUMMARY (Mean +/- Std across seeds)")
    print("="*70)
    print(summary.to_string(index=False))

    per_subject = results_df.groupby(['model', 'held_out']).agg({
        'accuracy': 'mean',
        'macro_f1': 'mean',
        'balanced_accuracy': 'mean'
    }).reset_index()
    per_subject_csv = os.path.join(RESULTS_DIR, "per_subject_performance.csv")
    per_subject.to_csv(per_subject_csv, index=False)
    print(f"Saved: {per_subject_csv}")

    print("\nDONE!")
    return results_df, summary

if __name__ == '__main__':
    results_df, summary = main()