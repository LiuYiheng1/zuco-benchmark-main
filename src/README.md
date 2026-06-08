# Environment Setup

## Requirements
```
scikit-learn>=1.0.2
numpy>=1.21.6
pandas>=1.3.5
torch>=1.9.0
```

## Installation
```bash
pip install -r requirements.txt
```

# Quick Start

## 1. Run Official Baseline (SVM)
```bash
cd src
python run_official_baseline.py
```

## 2. Run PyTorch Baselines
```bash
cd src/neuro_web_reader
python train_tgcr.py --model baselines --seeds 0 1 2 3 4
```

## 3. Run TGCR v1
```bash
python train_tgcr.py --model tgcr --seeds 0 1 2 3 4
```

## 4. Run All Experiments (including ablation)
```bash
python train_tgcr.py --model all --seeds 0 1 2 3 4
```

## 5. Generate Reports
```bash
python evaluate.py --results_dir results
```

# Models Available

## Baseline Models
- `eeg_mlp`: EEG-only MLP
- `gaze_mlp`: Gaze-only MLP
- `early_concat`: Early fusion (concat EEG + gaze, then MLP)
- `late_fusion`: Late fusion (separate encoders, merge logits)
- `attention_fusion`: Attention-based multimodal fusion

## TGCR Models
- `tgcr`: Full TGCR v1 with router
- `tgcr_no_router`: TGCR without router (simple concat fusion)
- `tgcr_eeg_only`: TGCR with EEG expert only
- `tgcr_gaze_only`: TGCR with gaze expert only
- `tgcr_random_router`: TGCR with random fixed router weights
- `tgcr_shuffle_eeg`: TGCR with shuffled EEG (control)
- `tgcr_shuffle_gaze`: TGCR with shuffled gaze (control)

# Output Files

- `results/`: Experiment results
  - `official_baseline_results_*.csv`: Official SVM baseline results
  - `all_results_*.csv`: All model results per sample
  - `summary_all_models_*.csv`: Summary statistics
  - `tgcr_router_weights_*.csv`: Router weights for analysis

- `reports/`: Generated reports
  - `baseline_reproduction.md`: Official baseline reproduction report
  - `tgcr_experiment_report_*.md`: TGCR experiment report