# AdaGTCN-style Debug Report

## Important Note
**This is a diagnostic experiment for cross-subject transfer analysis, NOT a main paper result.**

## 1. Subject Split (Single Split, seed=0)
- **Train**: ['YAG', 'YFS', 'YIS', 'YLS', 'YSD', 'YDR', 'YAK', 'YSL', 'YMD', 'YHS', 'YTL', 'YRK']
- **Validation**: ['YDG', 'YAC']
- **Test**: ['YFR', 'YRP']

## 2. Class Distribution
| Set | NR (label=1) | TSR (label=0) | NR Ratio |
|-----|--------------|----------------|----------|
| Train | 3257 | 3875 | 45.7% |
| Test | 368 | 369 | 49.9% |

**Label mapping**: NR=1, TSR=0 (consistent between train and test)

## 3. Baselines
- **Majority baseline**: 50.1% (predicts all as class 0)
- **Random baseline**: ~50%

## 4. Individual Model Results (Single Split)

| Method | Accuracy | Macro-F1 | Inverted Acc |
|--------|----------|----------|-------------|
| Majority | 50.1% | - | - |
| Random | ~50.0% | ~25% | - |
| EEG_SVM | 47.1% | 37.1% | 53.7% |
| Gaze_SVM | 56.3% | 55.6% | 53.6% |
| PCET_source | 46.3% | 39.8% | 53.7% |
| GETA_source | 46.4% | 36.7% | 53.6% |
| **PCET+GETA+CAGF** | **45.9%** | **40.0%** | 54.1% |

## 5. Confusion Matrices

### PCET
|  | Pred_TSR | Pred_NR |
|--|----------|---------|
| **Actual_TSR** | 50 | 319 |
| **Actual_NR** | 77 | 291 |

### GETA
|  | Pred_TSR | Pred_NR |
|--|----------|---------|
| **Actual_TSR** | 27 | 342 |
| **Actual_NR** | 53 | 315 |

### CAGF
|  | Pred_TSR | Pred_NR |
|--|----------|---------|
| **Actual_TSR** | 54 | 315 |
| **Actual_NR** | 84 | 284 |

## 6. CAGF Alpha Analysis
- **Mean**: 0.5147
- **Std**: 0.0691
- **Min**: 0.2689
- **Max**: 0.7311
- **Alpha < 0.3**: 11 (1.5%)
- **Alpha > 0.7**: 58 (7.9%)
- **Alpha in [0.3, 0.7]**: 668 (90.6%)

## 7. Label Inversion Test
If we invert all predictions:
- PCET: 53.7% (vs original 46.3%)
- GETA: 53.6% (vs original 46.4%)
- CAGF: 54.1% (vs original 45.9%)

**WARNING: Inverted accuracy is higher for some models! Possible label mapping issue.**

## 8. z_pcet vs z_geta Analysis
- z_pcet mean: 0.5692
- z_pcet std: 0.6582
- z_geta mean: 0.7492
- z_geta std: 0.6552
- z_pcet - z_geta mean: -0.1800

## 9. Key Findings

### Is 45.9% below majority baseline?
YES - the model is performing worse than majority!

### Does label inversion improve results?
YES - there may be a label mapping issue between train and test.

### Is CAGF alpha collapsed?
NO - alpha shows reasonable variation.

### Is cross-subject transfer working?
The accuracy of all models (45-56%) being close to or below majority (~50%) suggests that:
1. Cross-subject transfer is inherently difficult for this task
2. The EEG/gaze features may not generalize well across subjects
3. Subject-specific calibration (few-shot) would likely improve results significantly

## 10. Conclusions

- This is a **diagnostic** experiment, not a main paper result
- Zero-shot cross-subject performance is around random chance level
- Few-shot personalized calibration is likely essential for this approach
- The CAGF fusion does not provide significant improvement over individual models in zero-shot setting
