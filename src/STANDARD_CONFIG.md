# Standard Experiment Configuration

## Dataset & Protocol
- **Dataset**: ZuCo 2.0
- **Subjects**: 16 Y-subjects (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL)
- **Protocol**: LOSO (Leave-One-Subject-Out) with k-shot calibration
- **Shot values**: k = 3, 5, 10, 20, 50
- **Seeds**: 0, 1, 2, 3, 4

## Model: PCET + GETA + CAGF

### PCET (Prediction-error EEG Representation)
- Input: Raw EEG features (electrode_features_all.npy)
- PCA fitted on calibration data per class
- Error features: L2 norm of reconstruction error per class
- Classifier: SVC(kernel='rbf', probability=True)

### GETA (Gaze-guided EEG Task Attention)
- Gaze features: sent_gaze_sacc.npy
- Gaze MLP: hidden_layer_sizes=(32,)
- Attention: entropy * 0.01 + confidence
- EEG MLP: hidden_layer_sizes=(64, 32)

### CAGF (Cross-modal Adaptive Gated Fusion)
- Gate: alpha = sigmoid(z_eeg[:,0] - z_gaze[:,0])
- Fusion: z_fused = alpha * z_eeg + (1-alpha) * z_gaze
- Final MLP: hidden_layer_sizes=(16,)
- **NO confidence features used** (feature-only version)

## Reference Results

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 62.27% | 59.54% | 60.89% | 60.89% |
| 5 | 65.84% | 63.57% | 64.69% | 64.69% |
| 10 | 69.68% | 68.07% | 68.56% | 68.56% |
| 20 | 74.06% | 73.10% | 73.32% | 73.32% |
| 50 | 80.11% | 79.61% | 79.56% | 79.56% |

## File Paths
- Results: `results/final/eeg_gaze_pilot_results.csv`
- Config: `src/experiment_config.py`

## Key Rules
1. All future experiments must use **SAME model configuration** (SVC + MLP)
2. All future experiments must use **SAME protocol** (LOSO, k-shot)
3. All future experiments must use **SAME CAGF implementation** (feature-only, no confidence)
4. Report results in format: Accuracy ± std across subjects and seeds

## Important Note
This is the **BEST performing** version. Any new experiments should match this configuration exactly.