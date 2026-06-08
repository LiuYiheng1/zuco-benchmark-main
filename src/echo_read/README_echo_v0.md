# ECHO-Read v0

ECHO-Read: EEG and Gaze Cross-modal Observer for Reading Comprehension Analysis

## Overview

ECHO-Read is a cross-modal learning framework for analyzing reading comprehension using EEG and eye-tracking (gaze) data. The core idea is that EEG and Gaze are viewed as two coupled observations of the same underlying reading cognitive state.

## Framework Architecture

```
EEG (420-D) ──→ EEGObserver ──→ z_e ──┐
                                        │
                                        ▼
                              CommonCauseEstimator ──→ z_c ──→ Classifier ──→ NR/TSR
                                        │
                                        ▲
Gaze (9-D) ────→ GazeObserver ──→ z_g ──┘

z_c ──→ GazeDecoder ──→ gaze_hat_from_zc
z_c ──→ EEGDecoder ──→ eeg_hat_from_zc
z_e ──→ GazeDecoder ──→ gaze_hat_from_e
z_g ──→ EEGDecoder ──→ eeg_hat_from_g
```

## Components

### 1. EEGObserver
- Input: 420-D EEG features
- Architecture: 420 → 256 → 128 → d (default d=32)
- Output: z_e (EEG latent representation)

### 2. GazeObserver
- Input: 9-D gaze features
- Architecture: 9 → 64 → 64 → d (default d=32)
- Output: z_g (gaze latent representation)

### 3. CommonCauseEstimator
- Input: z_e, z_g
- Construction: h = [z_e, z_g, |z_e-z_g|, z_e*z_g]
- Architecture: 128 → d
- Output: z_c (common-cause latent reading state)

### 4. Decoders
- GazeDecoder: z → 64 → 9
- EEGDecoder: z → 128 → 420

### 5. Classifier
- Input: z_c
- Architecture: d → 64 → 2
- Output: logits for NR/TSR classification

## Loss Function

```
L = L_cls 
  + lambda_recon * L_common_recon 
  + lambda_cross * L_cross_pred 
  + lambda_align * L_latent_align
```

Where:
- L_cls = CrossEntropy(logits, y)
- L_common_recon = MSE(gaze_hat_from_zc, gaze) + alpha_eeg * MSE(eeg_hat_from_zc, eeg)
- L_cross_pred = MSE(gaze_hat_from_e, gaze) + alpha_eeg * MSE(eeg_hat_from_g, eeg)
- L_latent_align = SmoothL1(z_e, z_g)

Default parameters:
- lambda_recon = 0.05
- lambda_cross = 0.05
- lambda_align = 0.01
- alpha_eeg = 0.05

## Running Smoke Test

```bash
cd zuco-benchmark-main
python src/echo_read/train_echo_v0.py
```

This will run 4 modes:
1. gaze_mlp - Gaze-only classification baseline
2. eeg_mlp - EEG-only classification baseline  
3. concat_mlp - EEG+Gaze direct concatenation
4. echo_v0 - Full ECHO-Read model

## Output Files

- `results/echo_v0/smoke_test_log.txt` - Training log
- `results/echo_v0/smoke_test_metrics.csv` - Final metrics for all modes
- `results/echo_v0/smoke_test_shapes.json` - Data shape information
- `results/echo_v0/echo_v0_model_summary.txt` - Model parameter summary
- `results/echo_v0/protocol_checklist.md` - Protocol verification checklist

## Requirements

- Python 3.8+
- PyTorch 1.10+
- scikit-learn
- pandas
- numpy
- tqdm
- PyYAML

## Data Format

The framework expects aligned multimodal data in `data/aligned_multimodal_y.npz`:
- eeg: [N, 420] - EEG features
- gaze: [N, 9] - Gaze features
- y: [N] - Labels (0=TSR, 1=NR)

And corresponding metadata in `data/aligned_multimodal_y_metadata.csv`:
- sample_id, subject, label, idx, eeg_fullidx, gaze_fullidx, y, split

## Notes

- This is v0 (minimum viable implementation)
- No LLM integration
- No Agent components
- For smoke testing only, not optimized for performance