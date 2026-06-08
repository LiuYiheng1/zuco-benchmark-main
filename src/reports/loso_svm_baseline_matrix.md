# ZuCo 2.0 LOSO-Y Baseline Matrix Report

## Date: 2026-05-08

## Protocol
- **Method**: Leave-One-Subject-Out on Y-subjects
- **Folds**: 16 (one hold-out per Y-subject per seed)
- **Seeds**: [0, 1, 2, 3, 4]
- **Classifier**: SGDClassifier (hinge loss = linear SVM)
- **Scaler**: MinMaxScaler (feature_range=(0, 1)) - aligned with official validation.py
- **Shuffle**: Yes (train data shuffled before training) - aligned with official validation.py
- **X-subjects**: Excluded from local evaluation (hidden test)

## Settings vs Official validation.py

| Setting | Official | Ours | Status |
|---------|----------|------|--------|
| Subject list | 16 Y-subjects | 16 Y-subjects | OK |
| Label NR=1, TSR=0 | Yes | Yes | OK |
| Kernel | linear | linear (SGD hinge) | OK |
| SVM gamma | scale | scale | OK |
| Scaler | MinMaxScaler(0,1) | MinMaxScaler(0,1) | OK |
| Shuffle train | Yes | Yes | OK |
| Default seed | 1 | 0,1,2,3,4 | OK (extended) |

## Results Summary

### Overall Performance (Mean ± Std across 16 folds × 5 seeds)

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|-------|----------|-----------|------------------|
| Majority | 0.5399 ± 0.0321 | 0.3503 ± 0.0135 | 0.5000 ± 0.0000 |
| Random | 0.5023 ± 0.0216 | 0.4987 ± 0.0216 | 0.4994 ± 0.0215 |
| SVM_EEG_only | 0.5219 ± 0.0631 | 0.4449 ± 0.1013 | 0.5272 ± 0.0511 |
| SVM_Gaze_only | 0.6072 ± 0.1413 | 0.5583 ± 0.1720 | 0.6095 ± 0.1390 |
| SVM_Combined | 0.5943 ± 0.1190 | 0.5396 ± 0.1506 | 0.5987 ± 0.1175 |

## Key Findings

### 1. EEG-only vs Majority/Random

- EEG accuracy (0.5219) vs Majority (0.5399): -0.0180
- EEG accuracy (0.5219) vs Random (0.5023): +0.0196
- **EEG signal is NOT clearly above majority baseline** (slightly below)
- **EEG signal IS above random baseline** (by ~2%)

### 2. Gaze-only Performance

- Gaze accuracy (0.6072) vs EEG accuracy (0.5219): +0.0853
- **Gaze signal is STRONGER than EEG** (by ~8.5%)
- Gaze has very high subject variability (std = 0.1413)

### 3. Combined (EEG+Gaze) Performance

- Combined accuracy (0.5943) vs EEG (0.5219) vs Gaze (0.6072)
- **Combined does NOT significantly outperform gaze-only** (slightly lower)
- Fusion does NOT help when gaze alone is already the stronger modality

### 4. Subject Variability

Worst EEG subjects (by accuracy across seeds):
- YLS: ~0.41 (consistently lowest)
- YAC: ~0.44 (consistently low)
- YAK: ~0.51

Best EEG subjects (by accuracy across seeds):
- YMD: ~0.64 (consistently highest)
- YDG: ~0.63
- YIS: ~0.56

Worst Gaze subjects:
- YAK: ~0.42 (YAK gaze features may be noisy)
- YRP: ~0.47
- YAG: ~0.50

Best Gaze subjects:
- YTL: ~0.92 (extremely high, possible overfitting or unique pattern)
- YIS: ~0.86
- YSD: ~0.85

### 5. Macro-F1 Analysis

- Majority Macro-F1 (0.35) is deceptively low because it always predicts majority class
- Random Macro-F1 (~0.50) correctly reflects near-chance performance
- SVM models have variable Macro-F1 depending on per-class balance per subject

### 6. Extreme Subject Performance (Gaze-only)

Some subjects show extreme performance with gaze:
- YTL: 92.56% (5 seeds avg) - possibly unique gaze pattern
- YIS: 86.2% - very consistent across seeds
- YSD: 85.8% - very consistent across seeds

This suggests gaze patterns for these subjects are highly distinguishable between NR and TSR.

## Files Generated

- `svm_all_features_loso_20260508_173953.csv` - Full results (all folds, seeds)
- `summary_mean_std_20260508_173953.csv` - Summary statistics

## Conclusion

1. **EEG-only is at/near chance level**: 52.19% accuracy is marginally above random (50.23%) but below majority (53.99%)
2. **Gaze-only is the strongest signal**: 60.72% accuracy with high variability
3. **Combined does NOT outperform gaze-only**: Fusion does not help when gaze dominates
4. **Extreme subject variability**: Some subjects (YTL, YIS, YSD) have near-perfect gaze classification, while others (YAK, YRP) are at chance
5. **EEG signal is weak for cross-subject classification**: This confirms the original observation that EEG-only is near random
6. **These results are consistent with the official validation.py settings**: Same subjects, same label mapping, same scaler

## Next Steps

Before proceeding to TGCR or complex models, verify:
1. Why YTL/YIS/YSD gaze performance is so high (data quality or real signal?)
2. Why YAK/YRP gaze performance is so low
3. Whether EEG feature quality varies significantly across subjects
4. Whether the high variance in gaze-only is due to train/test distribution mismatch for certain subjects