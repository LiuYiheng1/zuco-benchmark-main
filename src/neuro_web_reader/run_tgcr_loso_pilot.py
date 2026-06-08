"""
TGCR v1 LOSO-Y Pilot Experiment
Tests TGCR and variants on Y-subject Leave-One-Subject-Out protocol
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.feature_loader import FeatureLoader
from models.tgcr import (
    TGCRv1, TGCRv1WithoutRouter, TGCRv1EEGonlY, TGCRv1GazeOnly, TGCRv1RandomRouter
)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "loso")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PairedDataset(Dataset):
    def __init__(self, eeg_X, gaze_X, y, meta=None):
        self.eeg = torch.FloatTensor(eeg_X)
        self.gaze = torch.FloatTensor(gaze_X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta is not None else [{}] * len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'eeg': self.eeg[idx], 'gaze': self.gaze[idx], 'label': self.y[idx], 'meta': self.meta[idx]}

class SingleDataset(Dataset):
    def __init__(self, X, y, meta=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta is not None else [{}] * len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'features': self.X[idx], 'label': self.y[idx], 'meta': self.meta[idx]}

def get_model(model_name, eeg_dim, gaze_dim, device):
    if model_name == 'tgcr_full':
        return TGCRv1(eeg_dim, gaze_dim).to(device)
    elif model_name == 'tgcr_no_router':
        return TGCRv1WithoutRouter(eeg_dim, gaze_dim).to(device)
    elif model_name == 'tgcr_random_router':
        return TGCRv1RandomRouter(eeg_dim, gaze_dim).to(device)
    elif model_name == 'tgcr_eeg_only':
        return TGCRv1EEGonlY(eeg_dim).to(device)
    elif model_name == 'tgcr_gaze_only':
        return TGCRv1GazeOnly(gaze_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, val_loader, epochs, lr, device, is_single=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if is_single:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                model_class_name = model.__class__.__name__
                if model_class_name == 'TGCRv1GazeOnly':
                    outputs = model(eeg=None, gaze=features).squeeze()
                else:
                    outputs = model(features).squeeze()
            else:
                eeg = batch['eeg'].to(device)
                gaze = batch['gaze'].to(device)
                labels = batch['label'].to(device)
                outputs = model(eeg, gaze).squeeze()

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_single:
                    features = batch['features'].to(device)
                    labels = batch['label']
                    model_class_name = model.__class__.__name__
                    if model_class_name == 'TGCRv1GazeOnly':
                        outputs = model(eeg=None, gaze=features).squeeze()
                    else:
                        outputs = model(features).squeeze()
                else:
                    eeg = batch['eeg'].to(device)
                    gaze = batch['gaze'].to(device)
                    labels = batch['label']
                    outputs = model(eeg, gaze).squeeze()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_fusion_model(model, test_loader, device):
    model.eval()
    predictions, probabilities, labels_list, router_weights_list = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            gaze = batch['gaze'].to(device)
            labels = batch['label']

            if hasattr(model, 'router') and not isinstance(model, (TGCRv1WithoutRouter, TGCRv1RandomRouter)):
                logits, r_weights = model(eeg, gaze, return_router_weights=True)
                router_weights_list.extend(r_weights.cpu().numpy())
            else:
                logits = model(eeg, gaze)

            probs = torch.sigmoid(logits.squeeze() if logits.dim() > 0 else logits)
            preds = (probs > 0.5).float()

            predictions.extend(preds.cpu().numpy() if preds.dim() > 0 else [preds.item()])
            probabilities.extend(probs.cpu().numpy() if probs.dim() > 0 else [probs.item()])
            labels_list.extend(labels.numpy())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels_array = np.array(labels_list)

    acc = accuracy_score(labels_array, predictions)
    f1 = f1_score(labels_array, predictions, average='macro')
    bacc = balanced_accuracy_score(labels_array, predictions)
    prec, rec, _, _ = precision_recall_fscore_support(labels_array, predictions, average='macro', warn_for=[])
    cm = confusion_matrix(labels_array, predictions)

    try:
        auroc = roc_auc_score(labels_array, probabilities)
    except:
        auroc = 0.5

    return {
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'precision_macro': prec,
        'recall_macro': rec,
        'auroc': auroc,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels_array
    }, router_weights_list

def evaluate_single_model(model, test_X, test_y, device, is_eeg_only=True):
    model.eval()
    X_t = torch.FloatTensor(test_X).to(device)

    with torch.no_grad():
        if is_eeg_only:
            logits = model(X_t).squeeze()
        else:
            logits = model(eeg=None, gaze=X_t).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    predictions = preds.cpu().numpy() if preds.dim() > 0 else np.array([preds.item()])
    probabilities = probs.cpu().numpy() if probs.dim() > 0 else np.array([probs.item()])
    labels_array = test_y

    acc = accuracy_score(labels_array, predictions)
    f1 = f1_score(labels_array, predictions, average='macro')
    bacc = balanced_accuracy_score(labels_array, predictions)
    prec, rec, _, _ = precision_recall_fscore_support(labels_array, predictions, average='macro', warn_for=[])
    cm = confusion_matrix(labels_array, predictions)

    try:
        auroc = roc_auc_score(labels_array, probabilities)
    except:
        auroc = 0.5

    return {
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'precision_macro': prec,
        'recall_macro': rec,
        'auroc': auroc,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels_array
    }

def load_loso_data(held_out_subject):
    loader = FeatureLoader(FEATURES_DIR)
    train_subjects = [s for s in Y_SUBJECTS if s != held_out_subject]

    eeg_X_train, gaze_X_train, y_train, _, meta_train = loader.load_eeg_gaze_paired_labeled(train_subjects)
    eeg_X_test, gaze_X_test, y_test, _, meta_test = loader.load_eeg_gaze_paired_labeled([held_out_subject])

    return (eeg_X_train, gaze_X_train, y_train, meta_train), (eeg_X_test, gaze_X_test, y_test, meta_test)

def run_tgcr_pilot(seed=0, epochs=50, batch_size=64, lr=0.001):
    print("="*70)
    print(f"TGCR v1 LOSO-Y Pilot (seed={seed})")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    models_to_test = [
        'tgcr_full',
        'tgcr_no_router',
        'tgcr_random_router',
        'tgcr_eeg_only',
        'tgcr_gaze_only',
        'tgcr_shuffle_eeg',
        'tgcr_shuffle_gaze'
    ]

    all_results = []
    all_predictions = []
    all_router_weights = []

    for held_out in Y_SUBJECTS:
        print(f"\n--- Held-out: {held_out} ---")

        (eeg_X_train, gaze_X_train, y_train, meta_train), (eeg_X_test, gaze_X_test, y_test, meta_test) = load_loso_data(held_out)

        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

        if len(eeg_X_train) == 0 or len(eeg_X_test) == 0:
            print(f"  Skipping {held_out} - no data")
            continue

        set_seed(seed)

        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        val_size = int(len(y_train) * 0.1)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        scaler_eeg = MinMaxScaler(feature_range=(0, 1))
        scaler_gaze = MinMaxScaler(feature_range=(0, 1))

        eeg_train_raw = eeg_X_train[train_idx]
        gaze_train_raw = gaze_X_train[train_idx]
        eeg_val_raw = eeg_X_train[val_idx]
        gaze_val_raw = gaze_X_train[val_idx]

        eeg_train = scaler_eeg.fit_transform(eeg_train_raw)
        gaze_train = scaler_gaze.fit_transform(gaze_train_raw)
        eeg_val = scaler_eeg.transform(eeg_val_raw)
        gaze_val = scaler_gaze.transform(gaze_val_raw)
        eeg_test = scaler_eeg.transform(eeg_X_test)
        gaze_test = scaler_gaze.transform(gaze_X_test)

        train_dataset = PairedDataset(eeg_train, gaze_train, y_train[train_idx], [meta_train[i] for i in train_idx])
        val_dataset = PairedDataset(eeg_val, gaze_val, y_train[val_idx], [meta_train[i] for i in val_idx])
        test_dataset = PairedDataset(eeg_test, gaze_test, y_test, meta_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        eeg_dim = eeg_train.shape[1]
        gaze_dim = gaze_train.shape[1]

        for model_name in models_to_test:
            actual_model_name = model_name
            shuffle_eeg = False
            shuffle_gaze = False

            if model_name == 'tgcr_shuffle_eeg':
                actual_model_name = 'tgcr_full'
                shuffle_eeg = True
            elif model_name == 'tgcr_shuffle_gaze':
                actual_model_name = 'tgcr_full'
                shuffle_gaze = True

            is_single = actual_model_name in ['tgcr_eeg_only', 'tgcr_gaze_only']

            eeg_train_use = eeg_train.copy()
            gaze_train_use = gaze_train.copy()
            y_train_use = y_train[train_idx]
            meta_train_use = [meta_train[i] for i in train_idx]

            if shuffle_eeg:
                idx = np.arange(len(eeg_train_use))
                np.random.shuffle(idx)
                eeg_train_use = eeg_train_use[idx]
                y_train_use = y_train_use[idx]
                meta_train_use = [meta_train_use[i] for i in idx]

            if shuffle_gaze:
                idx = np.arange(len(gaze_train_use))
                np.random.shuffle(idx)
                gaze_train_use = gaze_train_use[idx]
                y_train_use = y_train_use[idx]
                meta_train_use = [meta_train_use[i] for i in idx]

            if is_single:
                eeg_only = actual_model_name == 'tgcr_eeg_only'
                single_X = eeg_train_use if eeg_only else gaze_train_use
                single_y = y_train_use
                single_meta = meta_train_use
                val_X = eeg_val if eeg_only else gaze_val
                val_dataset = SingleDataset(val_X, y_train[val_idx], [meta_train[i] for i in val_idx])
                train_dataset = SingleDataset(single_X, single_y, single_meta)
            else:
                val_dataset = PairedDataset(eeg_val, gaze_val, y_train[val_idx], [meta_train[i] for i in val_idx])
                train_dataset = PairedDataset(eeg_train_use, gaze_train_use, y_train_use, meta_train_use)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = get_model(actual_model_name, eeg_dim, gaze_dim, device)
            model = train_model(model, train_loader, val_loader, epochs, lr, device, is_single=is_single)

            if is_single:
                eeg_only = actual_model_name == 'tgcr_eeg_only'
                test_X = eeg_test if eeg_only else gaze_test
                metrics = evaluate_single_model(model, test_X, y_test, device, is_eeg_only=eeg_only)
                router_weights = []
            else:
                metrics, router_weights = evaluate_fusion_model(model, test_loader, device)

            metrics['model'] = model_name
            metrics['seed'] = seed
            metrics['held_out'] = held_out
            all_results.append(metrics)

            predictions = metrics['predictions']
            probabilities = metrics['probabilities']

            print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

            for i, meta in enumerate(meta_test):
                all_predictions.append({
                    'model': model_name,
                    'seed': seed,
                    'held_out': held_out,
                    'subject_id': meta['subject_id'],
                    'sentence_id': meta['sentence_id'],
                    'full_idx': meta['full_idx'],
                    'true_label': int(y_test[i]),
                    'pred_label': int(predictions[i]) if i < len(predictions) else 0,
                    'pred_prob': float(probabilities[i]) if i < len(probabilities) else 0.5
                })

            if router_weights and model_name == 'tgcr_full':
                for i, meta in enumerate(meta_test):
                    w = router_weights[i] if i < len(router_weights) else [0, 0, 0, 0]
                    all_router_weights.append({
                        'model': model_name,
                        'seed': seed,
                        'held_out': held_out,
                        'subject_id': meta['subject_id'],
                        'sentence_id': meta['sentence_id'],
                        'full_idx': meta['full_idx'],
                        'true_label': int(y_test[i]),
                        'router_weight_eeg': float(w[0]),
                        'router_weight_gaze': float(w[1]),
                        'router_weight_fusion': float(w[2]) if len(w) > 2 else 0.0,
                        'router_weight_expert4': float(w[3]) if len(w) > 3 else 0.0
                    })

    return all_results, all_predictions, all_router_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    print(f"Starting TGCR pilot with seed={args.seed}")

    all_results, all_predictions, all_router_weights = run_tgcr_pilot(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(all_results)
    for col in ['predictions', 'probabilities', 'labels', 'confusion_matrix']:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])
    results_csv = os.path.join(RESULTS_DIR, f"tgcr_pilot_seed{args.seed}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    predictions_df = pd.DataFrame(all_predictions)
    predictions_csv = os.path.join(RESULTS_DIR, f"tgcr_predictions_seed{args.seed}.csv")
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Saved: {predictions_csv}")

    if all_router_weights:
        router_df = pd.DataFrame(all_router_weights)
        router_csv = os.path.join(RESULTS_DIR, f"tgcr_router_weights_seed{args.seed}.csv")
        router_df.to_csv(router_csv, index=False)
        print(f"Saved: {router_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    summary.columns = ['model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'balanced_accuracy_mean', 'balanced_accuracy_std']

    summary_csv = os.path.join(RESULTS_DIR, f"tgcr_summary_seed{args.seed}.csv")
    summary.to_csv(summary_csv, index=False)

    print("\n" + "="*70)
    print("TGCR PILOT SUMMARY (Seed={})".format(args.seed))
    print("="*70)
    for _, row in summary.iterrows():
        print(f"{row['model']:25s}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} | F1={row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()