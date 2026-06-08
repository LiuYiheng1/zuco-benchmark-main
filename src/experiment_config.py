"""
Standard Experiment Configuration
Based on the best performing version: PCET + GETA + CAGF

This configuration should be used for all future experiments.
"""

import os

EXPERIMENT_CONFIG = {
    # Dataset
    "dataset": "ZuCo 2.0",
    "subjects": ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL'],
    "n_subjects": 16,

    # Protocol
    "protocol": "LOSO (Leave-One-Subject-Out)",
    "shot_values": [3, 5, 10, 20, 50],
    "seeds": [0, 1, 2, 3, 4],

    # Evaluation
    "metrics": ["Accuracy", "Macro-F1", "BAcc", "AUROC"],

    # Model: PCET + GETA + CAGF
    "model": {
        "name": "PCET + GETA + CAGF",
        "description": """
        PCET: EEG prediction-error representation using PCA reconstruction error
        GETA: Gaze-guided attention on EEG features
        CAGF: Cross-modal Adaptive Gated Fusion (feature-only, no confidence)
        """,
        "key_features": [
            "SVC with RBF kernel for EEG/Gaze classification",
            "MLP(64,32) for EEG features",
            "MLP(32,) for Gaze features",
            "MLP(16,) for final fusion layer",
            "alpha = sigmoid(z_eeg[:,0] - z_gaze[:,0]) for gating"
        ]
    },

    # Results (Reference)
    "reference_results": {
        "k=3": {"Accuracy": 62.27, "Macro-F1": 59.54, "BAcc": 60.89, "AUROC": 60.89},
        "k=5": {"Accuracy": 65.84, "Macro-F1": 63.57, "BAcc": 64.69, "AUROC": 64.69},
        "k=10": {"Accuracy": 69.68, "Macro-F1": 68.07, "BAcc": 68.56, "AUROC": 68.56},
        "k=20": {"Accuracy": 74.06, "Macro-F1": 73.10, "BAcc": 73.32, "AUROC": 73.32},
        "k=50": {"Accuracy": 80.11, "Macro-F1": 79.61, "BAcc": 79.56, "AUROC": 79.56},
    },

    # File paths
    "data_dir": "features",
    "results_dir": "results/final",
    "reports_dir": "reports/final",
    "results_file": "results/final/eeg_gaze_pilot_results.csv",

    # Class labels
    "class_labels": {"TSR": 0, "NR": 1},
}

def print_config():
    print("="*80)
    print("STANDARD EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"\nDataset: {EXPERIMENT_CONFIG['dataset']}")
    print(f"Subjects: {EXPERIMENT_CONFIG['n_subjects']}")
    print(f"Protocol: {EXPERIMENT_CONFIG['protocol']}")
    print(f"Shot values: {EXPERIMENT_CONFIG['shot_values']}")
    print(f"Seeds: {EXPERIMENT_CONFIG['seeds']}")
    print(f"\nModel: {EXPERIMENT_CONFIG['model']['name']}")
    print(f"\nReference Results (k=50): {EXPERIMENT_CONFIG['reference_results']['k=50']}")

if __name__ == '__main__':
    print_config()