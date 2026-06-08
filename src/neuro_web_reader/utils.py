"""
Utility Functions for TGCR experiments
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import random

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc
    }

    if y_prob is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auroc'] = None

    return metrics

def shuffle_train_eeg(eeg_X: np.ndarray, y: np.ndarray, meta: List[Dict], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    set_seed(seed)
    indices = np.arange(len(eeg_X))
    np.random.shuffle(indices)
    return eeg_X[indices], y[indices], [meta[i] for i in indices]

def shuffle_train_gaze(gaze_X: np.ndarray, y: np.ndarray, meta: List[Dict], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    set_seed(seed)
    indices = np.arange(len(gaze_X))
    np.random.shuffle(indices)
    return gaze_X[indices], y[indices], [meta[i] for i in indices]

def aggregate_by_subject(results_df) -> Dict:
    subject_stats = results_df.groupby('subject_id').agg({
        'accuracy': 'mean',
        'macro_f1': 'mean',
        'balanced_accuracy': 'mean'
    }).to_dict()

    return subject_stats

def format_results_table(summary_df) -> str:
    table = "| Model | Accuracy | Macro-F1 | Balanced Acc |\n"
    table += "|-------|----------|----------|---------------|\n"

    for _, row in summary_df.iterrows():
        acc_mean = row.get('accuracy_mean', row.get('accuracy', 0))
        acc_std = row.get('accuracy_std', 0)
        f1_mean = row.get('macro_f1_mean', row.get('macro_f1', 0))
        f1_std = row.get('macro_f1_std', 0)
        bacc_mean = row.get('balanced_accuracy_mean', row.get('balanced_accuracy', 0))
        bacc_std = row.get('balanced_accuracy_std', 0)

        model = row.get('model', 'Unknown')
        table += f"| {model} | {acc_mean:.4f} ± {acc_std:.4f} | {f1_mean:.4f} ± {f1_std:.4f} | {bacc_mean:.4f} ± {bacc_std:.4f} |\n"

    return table

def analyze_router_weights(router_df) -> Dict:
    if router_df is None or len(router_df) == 0:
        return {}

    stats = {
        'eeg_weight_mean': router_df['router_weight_eeg'].mean(),
        'eeg_weight_std': router_df['router_weight_eeg'].std(),
        'gaze_weight_mean': router_df['router_weight_gaze'].mean(),
        'gaze_weight_std': router_df['router_weight_gaze'].std(),
        'fusion_weight_mean': router_df['router_weight_fusion'].mean(),
        'fusion_weight_std': router_df['router_weight_fusion'].std()
    }

    return stats