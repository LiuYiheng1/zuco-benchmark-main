"""
Complete Training Script for TGCR v1 and All Baselines with Ablation
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.feature_loader import FeatureLoader
from models.baselines import (
    EEGOnlyMLP, GazeOnlyMLP, EarlyConcatMLP, LateFusionModel, AttentionFusion
)
from models.tgcr import (
    TGCRv1, TGCRv1WithoutRouter, TGCRv1EEGonlY, TGCRv1GazeOnly, TGCRv1RandomRouter
)


class TGCRDataset(Dataset):
    def __init__(self, eeg_X: np.ndarray, gaze_X: np.ndarray, y: np.ndarray, meta: List[Dict] = None):
        self.eeg_X = torch.FloatTensor(eeg_X)
        self.gaze_X = torch.FloatTensor(gaze_X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta is not None else [{}] * len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_X[idx],
            'gaze': self.gaze_X[idx],
            'label': self.y[idx],
            'meta': self.meta[idx]
        }


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, meta: List[Dict] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta is not None else [{}] * len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'features': self.X[idx], 'label': self.y[idx], 'meta': self.meta[idx]}


class GazeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, meta: List[Dict] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.meta = meta if meta is not None else [{}] * len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'features': self.X[idx], 'label': self.y[idx], 'meta': self.meta[idx]}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_name: str, eeg_dim: int, gaze_dim: int, device: torch.device):
    if model_name == 'eeg_mlp':
        return EEGOnlyMLP(eeg_dim).to(device)
    elif model_name == 'gaze_mlp':
        return GazeOnlyMLP(gaze_dim).to(device)
    elif model_name == 'early_concat':
        return EarlyConcatMLP(eeg_dim, gaze_dim).to(device)
    elif model_name == 'late_fusion':
        return LateFusionModel(eeg_dim, gaze_dim).to(device)
    elif model_name == 'attention_fusion':
        return AttentionFusion(eeg_dim, gaze_dim).to(device)
    elif model_name == 'tgcr':
        return TGCRv1(eeg_dim, gaze_dim).to(device)
    elif model_name == 'tgcr_no_router':
        return TGCRv1WithoutRouter(eeg_dim, gaze_dim).to(device)
    elif model_name == 'tgcr_eeg_only':
        return TGCRv1EEGonlY(eeg_dim).to(device)
    elif model_name == 'tgcr_gaze_only':
        return TGCRv1GazeOnly(gaze_dim).to(device)
    elif model_name == 'tgcr_random_router':
        return TGCRv1RandomRouter(eeg_dim, gaze_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_fusion_model(model, train_loader, val_loader, epochs: int, lr: float, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            gaze = batch['gaze'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(eeg, gaze).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                gaze = batch['gaze'].to(device)
                labels = batch['label']

                outputs = model(eeg, gaze).squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
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


def train_eeg_model(model, train_loader, val_loader, epochs: int, lr: float, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label']

                outputs = model(features).squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
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


def run_single_experiment(model_name: str, train_data: Tuple, test_data: Tuple,
                         seed: int, device: torch.device, epochs: int = 100,
                         batch_size: int = 64, lr: float = 0.001,
                         shuffle_eeg: bool = False, shuffle_gaze: bool = False) -> Tuple[Dict, Optional[Dict]]:
    set_seed(seed)

    eeg_X_train, gaze_X_train, y_train, meta_train = train_data
    eeg_X_test, gaze_X_test, y_test, meta_test = test_data

    if shuffle_eeg:
        indices = np.arange(len(eeg_X_train))
        np.random.shuffle(indices)
        eeg_X_train = eeg_X_train[indices]
        y_train = y_train[indices]
        meta_train = [meta_train[i] for i in indices]

    if shuffle_gaze:
        indices = np.arange(len(gaze_X_train))
        np.random.shuffle(indices)
        gaze_X_train = gaze_X_train[indices]
        y_train = y_train[indices]
        meta_train = [meta_train[i] for i in indices]

    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()

    eeg_X_train_scaled = scaler_eeg.fit_transform(eeg_X_train)
    eeg_X_test_scaled = scaler_eeg.transform(eeg_X_test)

    gaze_X_train_scaled = scaler_gaze.fit_transform(gaze_X_train)
    gaze_X_test_scaled = scaler_gaze.transform(gaze_X_test)

    is_eeg_only = model_name in ['eeg_mlp', 'tgcr_eeg_only']
    is_gaze_only = model_name in ['gaze_mlp', 'tgcr_gaze_only']

    if is_eeg_only:
        train_dataset = EEGDataset(eeg_X_train_scaled, y_train, meta_train)
        test_dataset = EEGDataset(eeg_X_test_scaled, y_test, meta_test)
    elif is_gaze_only:
        train_dataset = GazeDataset(gaze_X_train_scaled, y_train, meta_train)
        test_dataset = GazeDataset(gaze_X_test_scaled, y_test, meta_test)
    else:
        train_dataset = TGCRDataset(eeg_X_train_scaled, gaze_X_train_scaled, y_train, meta_train)
        test_dataset = TGCRDataset(eeg_X_test_scaled, gaze_X_test_scaled, y_test, meta_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, eeg_X_train.shape[1], gaze_X_train.shape[1], device)

    if is_eeg_only or is_gaze_only:
        model = train_eeg_model(model, train_loader, test_loader, epochs, lr, device)
    else:
        model = train_fusion_model(model, train_loader, test_loader, epochs, lr, device)

    model.eval()
    predictions = []
    probabilities = []
    labels_list = []
    router_weights_list = []

    with torch.no_grad():
        for batch in test_loader:
            if is_eeg_only:
                features = batch['features'].to(device)
                labels = batch['label']
                logits = model(features)
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
            elif is_gaze_only:
                features = batch['features'].to(device)
                labels = batch['label']
                logits = model(features)
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
            else:
                eeg = batch['eeg'].to(device)
                gaze = batch['gaze'].to(device)
                labels = batch['label']

                if model_name == 'tgcr':
                    logits, r_weights = model(eeg, gaze, return_router_weights=True)
                    router_weights_list.extend(r_weights.cpu().numpy())
                else:
                    logits = model(eeg, gaze)

                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

            labels_list.extend(labels.numpy())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels_array = np.array(labels_list)

    acc = accuracy_score(labels_array, predictions)
    f1 = f1_score(labels_array, predictions, average='macro')
    bacc = balanced_accuracy_score(labels_array, predictions)
    try:
        auroc = roc_auc_score(labels_array, probabilities)
    except:
        auroc = None

    results = {
        'model': model_name,
        'seed': seed,
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'auroc': auroc,
        'shuffle_eeg': shuffle_eeg,
        'shuffle_gaze': shuffle_gaze
    }

    sample_results = []
    for i, meta in enumerate(meta_test):
        sample_results.append({
            'model': model_name,
            'seed': seed,
            'subject_id': meta['subject_id'],
            'sentence_id': meta['sentence_id'],
            'full_idx': meta['full_idx'],
            'true_label': int(labels_array[i]),
            'pred_label': int(predictions[i]),
            'pred_prob': float(probabilities[i])
        })

    router_results = None
    if model_name == 'tgcr' and len(router_weights_list) > 0:
        router_results = []
        for i, meta in enumerate(meta_test):
            router_results.append({
                'model': model_name,
                'seed': seed,
                'subject_id': meta['subject_id'],
                'sentence_id': meta['sentence_id'],
                'full_idx': meta['full_idx'],
                'true_label': int(labels_array[i]),
                'router_weight_eeg': float(router_weights_list[i][0]),
                'router_weight_gaze': float(router_weights_list[i][1]),
                'router_weight_fusion': float(router_weights_list[i][2])
            })

    return results, sample_results, router_results


def run_model_experiments(model_name: str, train_data: Tuple, test_data: Tuple,
                         seeds: List[int], device: torch.device,
                         epochs: int, batch_size: int, lr: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_results = []
    all_sample_results = []
    all_router_results = []

    for seed in seeds:
        results, sample_results, router_results = run_single_experiment(
            model_name, train_data, test_data, seed, device, epochs, batch_size, lr
        )
        all_results.append(results)
        all_sample_results.extend(sample_results)
        if router_results:
            all_router_results.extend(router_results)

    return all_results, all_sample_results, all_router_results


def run_ablation_experiments(base_train_data: Tuple, base_test_data: Tuple,
                             seeds: List[int], device: torch.device,
                             epochs: int, batch_size: int, lr: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_results = []
    all_sample_results = []
    all_router_results = []

    ablation_models = [
        ('tgcr', False, False),
        ('tgcr_no_router', False, False),
        ('tgcr_eeg_only', False, False),
        ('tgcr_gaze_only', False, False),
        ('tgcr_random_router', False, False),
        ('tgcr', True, False),
        ('tgcr', False, True)
    ]

    for model_name, shuffle_eeg, shuffle_gaze in ablation_models:
        model_key = model_name
        if shuffle_eeg:
            model_key += '_shuffle_eeg'
        if shuffle_gaze:
            model_key += '_shuffle_gaze'

        for seed in seeds:
            set_seed(seed)

            eeg_X_train, gaze_X_train, y_train, meta_train = base_train_data
            eeg_X_test, gaze_X_test, y_test, meta_test = base_test_data

            if shuffle_eeg:
                indices = np.arange(len(eeg_X_train))
                np.random.shuffle(indices)
                eeg_X_train = eeg_X_train[indices]
                y_train = y_train[indices]
                meta_train = [meta_train[i] for i in indices]

            if shuffle_gaze:
                indices = np.arange(len(gaze_X_train))
                np.random.shuffle(indices)
                gaze_X_train = gaze_X_train[indices]
                y_train = y_train[indices]
                meta_train = [meta_train[i] for i in indices]

            train_data = (eeg_X_train, gaze_X_train, y_train, meta_train)
            test_data = (eeg_X_test, gaze_X_test, y_test, meta_test)

            results, sample_results, router_results = run_single_experiment(
                model_key, train_data, test_data, seed, device, epochs, batch_size, lr
            )
            all_results.append(results)
            all_sample_results.extend(sample_results)
            if router_results:
                all_router_results.extend(router_results)

            print(f"  {model_key} seed {seed}: Acc={results['accuracy']:.4f}, F1={results['macro_f1']:.4f}")

    return all_results, all_sample_results, all_router_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', help='Model to run: all, baselines, tgcr, ablation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--features_dir', type=str, default='features')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    src_dir = project_root if os.path.basename(project_root) == 'src' else os.path.join(project_root, 'src')
    features_dir = os.path.join(src_dir, args.features_dir)
    results_dir = os.path.join(src_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    loader = FeatureLoader(features_dir)

    train_subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
    test_subjects = ["XBB", "XDT", "XLS", "XPB", "XSE", "XTR", "XWS", "XAH", "XBD", "XSS"]

    eeg_X_train, gaze_X_train, y_train, meta_train = loader.load_eeg_gaze_paired(train_subjects)
    eeg_X_test, gaze_X_test, y_test, meta_test = loader.load_eeg_gaze_paired(test_subjects)

    print(f"Train: EEG {eeg_X_train.shape}, Gaze {gaze_X_train.shape}")
    print(f"Test: EEG {eeg_X_test.shape}, Gaze {gaze_X_test.shape}")

    train_data = (eeg_X_train.copy(), gaze_X_train.copy(), y_train.copy(), meta_train.copy())
    test_data = (eeg_X_test.copy(), gaze_X_test.copy(), y_test.copy(), meta_test.copy())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    all_sample_results = []
    all_router_results = []

    if args.model in ['all', 'baselines']:
        baseline_models = ['eeg_mlp', 'gaze_mlp', 'early_concat', 'late_fusion', 'attention_fusion']
        for model_name in baseline_models:
            print(f"\nTraining {model_name}...")
            results, sample_results, router_results = run_model_experiments(
                model_name, train_data, test_data,
                seeds=args.seeds, device=device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
            )
            all_results.extend(results)
            all_sample_results.extend(sample_results)
            if router_results:
                all_router_results.extend(router_results)

            seed_df = pd.DataFrame(results)
            print(f"  {model_name}: Acc={seed_df['accuracy'].mean():.4f}±{seed_df['accuracy'].std():.4f}, F1={seed_df['macro_f1'].mean():.4f}±{seed_df['macro_f1'].std():.4f}")

    if args.model in ['all', 'tgcr']:
        print(f"\nTraining TGCR...")
        results, sample_results, router_results = run_model_experiments(
            'tgcr', train_data, test_data,
            seeds=args.seeds, device=device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        all_results.extend(results)
        all_sample_results.extend(sample_results)
        if router_results:
            all_router_results.extend(router_results)

        seed_df = pd.DataFrame(results)
        print(f"  TGCR: Acc={seed_df['accuracy'].mean():.4f}±{seed_df['accuracy'].std():.4f}, F1={seed_df['macro_f1'].mean():.4f}±{seed_df['macro_f1'].std():.4f}")

    if args.model == 'ablation':
        print(f"\nRunning ablation experiments...")
        results, sample_results, router_results = run_ablation_experiments(
            train_data, test_data,
            seeds=args.seeds, device=device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        all_results.extend(results)
        all_sample_results.extend(sample_results)
        if router_results:
            all_router_results.extend(router_results)

    if args.model == 'all':
        ablation_models = [
            ('tgcr_no_router', False, False),
            ('tgcr_eeg_only', False, False),
            ('tgcr_gaze_only', False, False),
            ('tgcr_random_router', False, False),
            ('tgcr_shuffle_eeg', True, False),
            ('tgcr_shuffle_gaze', False, True)
        ]
        for model_name, shuffle_eeg, shuffle_gaze in ablation_models:
            print(f"\nTraining {model_name}...")
            for seed in args.seeds:
                set_seed(seed)

                eeg_X_t, gaze_X_t, y_t, meta_t = train_data[0].copy(), train_data[1].copy(), train_data[2].copy(), train_data[3].copy()
                eeg_X_te, gaze_X_te, y_te, meta_te = test_data[0].copy(), test_data[1].copy(), test_data[2].copy(), test_data[3].copy()

                if shuffle_eeg:
                    indices = np.arange(len(eeg_X_t))
                    np.random.shuffle(indices)
                    eeg_X_t = eeg_X_t[indices]
                    y_t = y_t[indices]
                    meta_t = [meta_t[i] for i in indices]

                if shuffle_gaze:
                    indices = np.arange(len(gaze_X_t))
                    np.random.shuffle(indices)
                    gaze_X_t = gaze_X_t[indices]
                    y_t = y_t[indices]
                    meta_t = [meta_t[i] for i in indices]

                t_data = (eeg_X_t, gaze_X_t, y_t, meta_t)
                te_data = (eeg_X_te, gaze_X_te, y_te, meta_te)

                results, sample_results, router_results = run_single_experiment(
                    model_name, t_data, te_data, seed, device, args.epochs, args.batch_size, args.lr
                )
                all_results.append(results)
                all_sample_results.extend(sample_results)
                if router_results:
                    all_router_results.extend(router_results)

                print(f"  {model_name} seed {seed}: Acc={results['accuracy']:.4f}, F1={results['macro_f1']:.4f}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = os.path.join(results_dir, f"all_results_{timestamp}.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\nSaved results to {results_csv}")

        sample_df = pd.DataFrame(all_sample_results)
        sample_csv = os.path.join(results_dir, f"all_samples_{timestamp}.csv")
        sample_df.to_csv(sample_csv, index=False)
        print(f"Saved samples to {sample_csv}")

        summary = results_df.groupby('model').agg({
            'accuracy': ['mean', 'std'],
            'macro_f1': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'auroc': ['mean', 'std']
        }).reset_index()
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary_csv = os.path.join(results_dir, f"summary_all_models_{timestamp}.csv")
        summary.to_csv(summary_csv, index=False)
        print(f"Saved summary to {summary_csv}")

    if all_router_results:
        router_df = pd.DataFrame(all_router_results)
        router_csv = os.path.join(results_dir, f"tgcr_router_weights_{timestamp}.csv")
        router_df.to_csv(router_csv, index=False)
        print(f"Saved router weights to {router_csv}")

    print("\nDone!")


if __name__ == '__main__':
    main()