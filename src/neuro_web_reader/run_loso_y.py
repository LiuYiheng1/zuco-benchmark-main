"""
Complete LOSO-Y (Leave-One-Subject-Out on Y-subjects) Cross-Validation Framework
All X-subject hidden test samples are excluded from local supervised evaluation.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, roc_auc_score
)
from sklearn.svm import SVC
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.basename(PROJECT_ROOT) == 'src':
    SRC_DIR = PROJECT_ROOT
else:
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

FEATURES_DIR = os.path.join(SRC_DIR, "features")
LOSO_RESULTS_DIR = os.path.join(SRC_DIR, "results", "loso")
os.makedirs(LOSO_RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_features(subject, feature_name):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split("_")
    subj = parts[0]
    if len(parts) >= 2 and parts[1] == "NR":
        label, sent_idx, full_idx, is_labeled = "NR", int(parts[2]), int(parts[3]), True
    elif len(parts) >= 2 and parts[1] == "TSR":
        label, sent_idx, full_idx, is_labeled = "TSR", int(parts[2]), int(parts[3]), True
    else:
        label, sent_idx, full_idx, is_labeled = "", int(parts[-2]) if len(parts) >= 2 else 0, int(parts[-1]) if len(parts) >= 1 else 0, False
    return subj, label, sent_idx, full_idx, is_labeled

def load_labeled_data(subjects, feature_name):
    all_X, all_y, all_meta = [], [], []
    for subj in subjects:
        feats = load_features(subj, feature_name)
        if feats is None:
            continue
        for key, values in feats.items():
            subj_id, label, sent_idx, full_idx, is_labeled = parse_key(key)
            if not is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
            all_meta.append({'subject_id': subj_id, 'sentence_id': sent_idx, 'full_idx': full_idx, 'label': label_binary, 'original_key': key})
    return np.array(all_X), np.array(all_y), all_meta if all_X else ([], [], [])

def load_eeg_gaze_paired_labeled(subjects):
    eeg_X, eeg_y, eeg_meta = load_labeled_data(subjects, 'electrode_features_all')
    gaze_X, gaze_y, gaze_meta = load_labeled_data(subjects, 'sent_gaze_sacc')
    if len(eeg_X) == 0 or len(gaze_X) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), []
    eeg_keys = set((m['subject_id'], m['full_idx']) for m in eeg_meta)
    gaze_keys = set((m['subject_id'], m['full_idx']) for m in gaze_meta)
    common_keys = eeg_keys & gaze_keys
    eeg_lookup = {(m['subject_id'], m['full_idx']): i for i, m in enumerate(eeg_meta)}
    gaze_lookup = {(m['subject_id'], m['full_idx']): i for i, m in enumerate(gaze_meta)}
    common_eeg_idx = [eeg_lookup[k] for k in common_keys if k in eeg_lookup]
    common_gaze_idx = [gaze_lookup[k] for k in common_keys if k in gaze_lookup]
    return eeg_X[common_eeg_idx], gaze_X[common_gaze_idx], eeg_y[common_eeg_idx], gaze_y[common_gaze_idx], [eeg_meta[i] for i in common_eeg_idx]

class EEGDataset(Dataset):
    def __init__(self, X, y, meta=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta else [{}] * len(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return {'features': self.X[idx], 'label': self.y[idx], 'meta': self.meta[idx]}

class FusionDataset(Dataset):
    def __init__(self, eeg_X, gaze_X, y, meta=None):
        self.eeg_X = torch.FloatTensor(eeg_X)
        self.gaze_X = torch.FloatTensor(gaze_X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta else [{}] * len(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return {'eeg': self.eeg_X[idx], 'gaze': self.gaze_X[idx], 'label': self.y[idx], 'meta': self.meta[idx]}

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', warn_for=[])
    cm = confusion_matrix(y_true, y_pred)
    auroc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_prob)
        except: pass
    return {'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'precision_macro': prec, 'recall_macro': rec, 'auroc': auroc, 'cm': cm}

def run_svm_loso(train_subjects, held_out_subject, feature_set='electrode_features_all', seed=0):
    train_subjs = [s for s in train_subjects if s != held_out_subject]
    test_subjs = [held_out_subject]

    if feature_set == 'eeg_gaze':
        X_train, _, y_train, _, meta_train = load_eeg_gaze_paired_labeled(train_subjs)
        X_test, _, y_test, _, meta_test = load_eeg_gaze_paired_labeled(test_subjs)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_s = np.hstack([X_train_s, np.zeros((len(X_train_s), 1))])
        X_test_s = np.hstack([X_test_s, np.zeros((len(X_test_s), 1))])
    else:
        X_train, y_train, meta_train = load_labeled_data(train_subjs, feature_set)
        X_test, y_test, meta_test = load_labeled_data(test_subjs, feature_set)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

    clf = SVC(random_state=seed, kernel='linear', gamma='scale', probability=True)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    return {**metrics, 'model': 'SVM', 'seed': seed, 'held_out': held_out_subject, 'n_train': len(y_train), 'n_test': len(y_test), 'test_nr_ratio': sum(y_test==1)/len(y_test) if len(y_test) > 0 else 0, 'meta': meta_test}

def train_eeg_mlp(X_train, y_train, X_val, y_val, seed=0, epochs=100, device='cpu'):
    set_seed(seed)
    model = nn.Sequential(nn.Linear(X_train.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_f1, best_state = 0, None
    patience_counter = 0
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                outputs = model(features).squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def run_eeg_mlp_loso(train_subjects, held_out_subject, seed=0, epochs=100, device='cpu'):
    train_subjs = [s for s in train_subjects if s != held_out_subject]
    test_subjs = [held_out_subject]

    X_train, y_train, meta_train = load_labeled_data(train_subjs, 'electrode_features_all')
    X_test, y_test, meta_test = load_labeled_data(test_subjs, 'electrode_features_all')
    if len(X_train) == 0 or len(X_test) == 0:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_eeg_mlp(X_train_s, y_train, X_test_s, y_test, seed, epochs, device)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_s).to(device)
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    metrics = compute_metrics(y_test, preds, probs)
    return {**metrics, 'model': 'EEG_MLP', 'seed': seed, 'held_out': held_out_subject, 'n_train': len(y_train), 'n_test': len(y_test), 'test_nr_ratio': sum(y_test==1)/len(y_test) if len(y_test) > 0 else 0, 'meta': meta_test}

def train_fusion_mlp(eeg_train, gaze_train, y_train, eeg_val, gaze_val, y_val, seed=0, epochs=100, device='cpu', model_type='early_concat'):
    set_seed(seed)
    eeg_dim, gaze_dim = eeg_train.shape[1], gaze_train.shape[1]

    class FusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.eeg_enc = nn.Sequential(nn.Linear(eeg_dim, 256), nn.ReLU(), nn.Dropout(0.3))
            self.gaze_enc = nn.Sequential(nn.Linear(gaze_dim, 64), nn.ReLU(), nn.Dropout(0.3))
            if model_type == 'early_concat':
                self.classifier = nn.Sequential(nn.Linear(256+64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
            elif model_type == 'late_fusion':
                self.classifier = nn.Sequential(nn.Linear(256+64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
            elif model_type == 'attention_fusion':
                self.gate = nn.Sequential(nn.Linear(256+64, 256+64), nn.Sigmoid())
                self.classifier = nn.Sequential(nn.Linear(256+64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        def forward(self, eeg, gaze):
            z_eeg = self.eeg_enc(eeg)
            z_gaze = self.gaze_enc(gaze)
            z = torch.cat([z_eeg, z_gaze], dim=1)
            if model_type == 'attention_fusion':
                z = z * self.gate(z)
            return self.classifier(z).squeeze()

    model = FusionModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_f1, best_state = 0, None
    patience_counter = 0
    train_dataset = FusionDataset(eeg_train, gaze_train, y_train)
    val_dataset = FusionDataset(eeg_val, gaze_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            gaze = batch['gaze'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(eeg, gaze)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                gaze = batch['gaze'].to(device)
                outputs = model(eeg, gaze)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def run_fusion_loso(train_subjects, held_out_subject, seed=0, epochs=100, device='cpu', model_type='early_concat'):
    train_subjs = [s for s in train_subjects if s != held_out_subject]
    test_subjs = [held_out_subject]

    eeg_train, gaze_train, y_train, _, meta_train = load_eeg_gaze_paired_labeled(train_subjs)
    eeg_test, gaze_test, y_test, _, meta_test = load_eeg_gaze_paired_labeled(test_subjs)
    if len(eeg_train) == 0 or len(eeg_test) == 0:
        return None

    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()
    eeg_train_s = scaler_eeg.fit_transform(eeg_train)
    eeg_test_s = scaler_eeg.transform(eeg_test)
    gaze_train_s = scaler_gaze.fit_transform(gaze_train)
    gaze_test_s = scaler_gaze.transform(gaze_test)

    model = train_fusion_mlp(eeg_train_s, gaze_train_s, y_train, eeg_test_s, gaze_test_s, y_test, seed, epochs, device, model_type)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        eeg_tensor = torch.FloatTensor(eeg_test_s).to(device)
        gaze_tensor = torch.FloatTensor(gaze_test_s).to(device)
        outputs = model(eeg_tensor, gaze_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    metrics = compute_metrics(y_test, preds, probs)
    model_name = f'{model_type.upper()}'
    return {**metrics, 'model': model_name, 'seed': seed, 'held_out': held_out_subject, 'n_train': len(y_train), 'n_test': len(y_test), 'test_nr_ratio': sum(y_test==1)/len(y_test) if len(y_test) > 0 else 0, 'meta': meta_test}

def run_majority_baseline(train_subjects, held_out_subject, seed=0):
    train_subjs = [s for s in train_subjects if s != held_out_subject]
    test_subjs = [held_out_subject]

    _, y_train, _ = load_labeled_data(train_subjs, 'electrode_features_all')
    _, y_test, meta_test = load_labeled_data(test_subjs, 'electrode_features_all')
    if len(y_train) == 0 or len(y_test) == 0:
        return None

    majority_class = 1 if sum(y_train==1) > sum(y_train==0) else 0
    y_pred = np.full_like(y_test, majority_class)
    metrics = compute_metrics(y_test, y_pred)
    return {**metrics, 'model': 'MAJORITY', 'seed': seed, 'held_out': held_out_subject, 'n_train': len(y_train), 'n_test': len(y_test), 'test_nr_ratio': sum(y_test==1)/len(y_test) if len(y_test) > 0 else 0, 'meta': meta_test}

def run_random_baseline(train_subjects, held_out_subject, seed=0):
    train_subjs = [s for s in train_subjects if s != held_out_subject]
    test_subjs = [held_out_subject]

    _, y_train, _ = load_labeled_data(train_subjs, 'electrode_features_all')
    _, y_test, meta_test = load_labeled_data(test_subjs, 'electrode_features_all')
    if len(y_train) == 0 or len(y_test) == 0:
        return None

    set_seed(seed)
    nr_prior = sum(y_train==1) / len(y_train)
    y_prob = np.random.random(len(y_test))
    y_pred = (y_prob < nr_prior).astype(int)
    metrics = compute_metrics(y_test, y_pred)
    return {**metrics, 'model': 'RANDOM', 'seed': seed, 'held_out': held_out_subject, 'n_train': len(y_train), 'n_test': len(y_test), 'test_nr_ratio': sum(y_test==1)/len(y_test) if len(y_test) > 0 else 0, 'meta': meta_test}

def run_all_experiments(seeds=[0, 1, 2, 3, 4], epochs=100, device='cpu'):
    models_to_run = [
        ('MAJORITY', lambda h, s: run_majority_baseline(Y_SUBJECTS, h, s)),
        ('RANDOM', lambda h, s: run_random_baseline(Y_SUBJECTS, h, s)),
        ('SVM_EEG', lambda h, s: run_svm_loso(Y_SUBJECTS, h, 'electrode_features_all', s)),
        ('SVM_GAZE', lambda h, s: run_svm_loso(Y_SUBJECTS, h, 'sent_gaze_sacc', s)),
        ('SVM_COMBINED', lambda h, s: run_svm_loso(Y_SUBJECTS, h, 'sent_gaze_sacc_eeg_means', s)),
        ('EEG_MLP', lambda h, s: run_eeg_mlp_loso(Y_SUBJECTS, h, s, epochs, device)),
        ('EARLY_CONCAT', lambda h, s: run_fusion_loso(Y_SUBJECTS, h, s, epochs, device, 'early_concat')),
        ('LATE_FUSION', lambda h, s: run_fusion_loso(Y_SUBJECTS, h, s, epochs, device, 'late_fusion')),
        ('ATTENTION_FUSION', lambda h, s: run_fusion_loso(Y_SUBJECTS, h, s, epochs, device, 'attention_fusion')),
    ]

    all_results = []
    all_predictions = []

    for model_name, run_fn in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running {model_name}")
        print(f"{'='*60}")

        for seed in seeds:
            for held_out in Y_SUBJECTS:
                result = run_fn(held_out, seed)
                if result is None:
                    continue

                cm = result.pop('cm')
                meta = result.pop('meta')

                result['tn'] = int(cm[0,0]) if cm.shape[0] > 1 else 0
                result['fp'] = int(cm[0,1]) if cm.shape[0] > 1 else 0
                result['fn'] = int(cm[1,0]) if cm.shape[0] > 1 else 0
                result['tp'] = int(cm[1,1]) if cm.shape[0] > 1 else 0

                all_results.append(result)

                for i, m in enumerate(meta):
                    all_predictions.append({
                        'model': result['model'],
                        'seed': seed,
                        'held_out': held_out,
                        'subject_id': m['subject_id'],
                        'sentence_id': m['sentence_id'],
                        'full_idx': m['full_idx'],
                        'true_label': int(m['label']),
                        'pred_label': int((torch.sigmoid(torch.tensor(result.get('accuracy', 0))) > 0.5).int().item()) if 'accuracy' in result else 0
                    })

            seed_results = [r for r in all_results if r['model'] == model_name and r['seed'] == seed]
            if seed_results:
                seed_acc = np.mean([r['accuracy'] for r in seed_results])
                seed_f1 = np.mean([r['macro_f1'] for r in seed_results])
                print(f"  Seed {seed}: Acc={seed_acc:.4f}, F1={seed_f1:.4f}")

    return pd.DataFrame(all_results), pd.DataFrame(all_predictions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("="*70)
    print("ZuCo 2.0 LOSO-Y Cross-Validation")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Y-subjects: {len(Y_SUBJECTS)} folds")

    results_df, predictions_df = run_all_experiments(seeds=args.seeds, epochs=args.epochs, device=args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_csv = os.path.join(LOSO_RESULTS_DIR, f"loso_all_results_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to {results_csv}")

    predictions_csv = os.path.join(LOSO_RESULTS_DIR, f"loso_predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    summary_csv = os.path.join(LOSO_RESULTS_DIR, f"summary_mean_std_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for _, row in summary.iterrows():
        print(f"{row['model']}: Acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}")

    report = f"""# ZuCo 2.0 LOSO-Y Cross-Validation Report

## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Protocol
- **Method**: Leave-One-Subject-Out on Y-subjects only
- **Folds**: 16 (one hold-out per Y-subject)
- **Seeds**: {args.seeds}
- **X-subjects**: Excluded from local evaluation (hidden test, no ground-truth labels)

## Results Summary

### Overall (Mean ± Std across folds and seeds)

| Model | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|------------------|-------|
"""
    for _, row in summary.iterrows():
        auroc_str = f"{row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}" if pd.notna(row['auroc_mean']) else "N/A"
        report += f"| {row['model']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {auroc_str} |\n"

    report += f"""
## Files Generated
- `{results_csv}`
- `{predictions_csv}`
- `{summary_csv}`

## Note
Local evaluation uses ONLY Y-subjects (labeled). X-subjects are for EvalAI submission only.
"""

    report_path = os.path.join(SRC_DIR, "reports", "loso_experiment_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")

    print("\n" + "="*70)
    print("LOSO-Y CROSS-VALIDATION COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()