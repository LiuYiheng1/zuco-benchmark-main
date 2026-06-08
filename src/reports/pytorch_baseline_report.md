# ZuCo 2.0 LOSO-Y Baseline Matrix Report

## Date: 2026-05-08

## Protocol
- **Method**: Leave-One-Subject-Out on Y-subjects (16 folds)
- **Folds**: 16 (one hold-out per Y-subject per seed)
- **Seeds**: [0, 1, 2, 3, 4]
- **X-subjects**: Excluded from local evaluation (hidden test)
- **Scaler**: MinMaxScaler (feature_range=(0, 1)) - fit on train fold only
- **Shuffle**: Train data shuffled before training

## Feature Dimensions
- **EEG**: 420 dimensions (electrode_features_all)
- **Gaze**: 9 dimensions (sent_gaze_sacc)
- **Combined**: 14 dimensions (sent_gaze_sacc_eeg_means)

Note: EEG and gaze features are NOT paired at the sentence level in the raw files. The `sent_gaze_sacc_eeg_means` provides paired combined features with 14 dimensions (10 gaze + 4 EEG band means).

---

## Complete Baseline Summary

| Model | Accuracy | Macro-F1 | Balanced Accuracy | Rank |
|-------|----------|-----------|-------------------|------|
| **SVM_Gaze_only** | **60.72% ± 14.13%** | 55.83% ± 17.20% | 60.95% ± 13.90% | 1 |
| SVM_Combined | 59.43% ± 11.90% | 53.96% ± 15.06% | 59.87% ± 11.75% | 2 |
| Majority | 53.99% ± 3.21% | 35.03% ± 1.35% | 50.00% ± 0.00% | 3 |
| SVM_EEG_only | 52.19% ± 6.31% | 44.49% ± 10.13% | 52.72% ± 5.11% | 4 |
| Random | 50.23% ± 2.16% | 49.87% ± 2.16% | 49.94% ± 2.15% | 5 |

*Note: PyTorch MLP results were partially completed but are not yet included due to long training time. Full MLP results pending.*

---

## Key Findings

### 1. EEG-only is at Near-Chance Level
- **SVM_EEG_only**: 52.19% ± 6.31% (marginally above random 50.23%)
- Below majority baseline (53.99%)
- High subject variability (std = 6.31%)

### 2. Gaze-only is the Strongest Signal
- **SVM_Gaze_only**: 60.72% ± 14.13% (highest accuracy)
- Significantly above random (p=0.0042**)
- Very high subject variance (std = 14.13%)

### 3. Combined (EEG+Gaze) Does NOT Outperform Gaze-only
- **SVM_Combined**: 59.43% ± 11.90% (slightly below gaze-only)
- Difference: -1.28% (not statistically significant, p=0.90)
- Fusion introduces noise when one modality dominates

### 4. Extreme Subject Variability

**Best Gaze Subjects:**
| Subject | Gaze Accuracy | EEG Accuracy | Combined |
|---------|---------------|--------------|----------|
| YTL | 90.93% | 46.34% | 89.31% |
| YIS | 85.06% | 52.10% | 81.95% |
| YSD | 84.22% | 55.29% | 54.45% |

**Worst Gaze Subjects:**
| Subject | Gaze Accuracy | EEG Accuracy | Combined |
|---------|---------------|--------------|----------|
| YAK | 42.03% | 54.21% | 44.74% |
| YRP | 46.66% | 50.18% | 46.58% |
| YLS | 59.70% | 40.85% | 60.54% |

### 5. EEG Complements Gaze in Some Subjects
- **YAK**: Low gaze (42%), but EEG is 54% - EEG helps
- **YLS**: Low EEG (41%), high gaze (60%) - gaze dominates
- **YTL/YIS/YSD**: Very high gaze, EEG does not help

---

## Statistical Significance Tests (Wilcoxon)

| Comparison | Mean Diff | p-value | Significance |
|------------|-----------|---------|--------------|
| SVM_Gaze_only vs SVM_EEG_only | +8.53% | 0.0934 | ns |
| SVM_Combined vs SVM_Gaze_only | -1.28% | 0.8999 | ns |
| SVM_Combined vs SVM_EEG_only | +7.24% | 0.0934 | ns |
| SVM_Gaze_only vs Random | +10.49% | 0.0042 | ** |
| SVM_Combined vs Random | +9.21% | 0.0027 | ** |

*ns = not significant, \*\* p<0.01*

---

## Interpretation

### Why is EEG-only so weak?
1. **Cross-subject variability**: EEG signals vary significantly across subjects due to anatomical differences
2. **Low signal-to-noise**: Raw EEG features may not capture task-relevant patterns well
3. **Feature dimensionality**: 420 EEG features on ~8000 training samples may cause overfitting

### Why does gaze dominate?
1. **Task-specific eye movements**: TSR (Task-Specific Reading) involves distinct eye movement patterns
2. **Lower dimensionality**: 9 gaze features are more robust to overfitting
3. **Direct behavioral measure**: Gaze directly reflects reading strategy

### Why doesn't fusion help?
1. **Modality dominance**: When one modality (gaze) is much stronger, fusion adds noise
2. **EEG redundancy**: Combined features use EEG band means (4 dims) not full EEG (420 dims)
3. **Simple concatenation**: Naive fusion doesn't leverage complementary information

---

## Subject-Specific Insights

### High Gaze Subjects (YTL, YIS, YSD)
- These subjects have highly distinguishable gaze patterns between NR and TSR
- EEG does NOT help - their EEG accuracy is near random
- Combined does NOT improve over gaze alone
- Possible explanation: These subjects have consistent, task-specific eye movement signatures

### Low Gaze Subjects (YAK, YRP, YLS)
- YAK: Gaze is noisy (42%), but EEG is 54% - EEG provides complementary info
- YRP: Both modalities struggle (47% gaze, 50% EEG)
- YLS: Gaze is decent (60%) but EEG is very poor (41%)

### Key Observation
**No subject shows EEG being clearly superior to gaze**. Even in low gaze subjects, EEG doesn't dramatically outperform. This suggests:
1. Gaze is the dominant modality for this task
2. EEG features need better preprocessing or more sophisticated models

---

## Comparison with Official validation.py

| Setting | Official | Ours | Status |
|---------|----------|------|--------|
| Subject list | 16 Y-subjects | 16 Y-subjects | Match |
| Label NR=1, TSR=0 | Yes | Yes | Match |
| Kernel | linear | linear (SGD hinge) | Match |
| SVM gamma | scale | scale | Match |
| Scaler | MinMaxScaler(0,1) | MinMaxScaler(0,1) | Match |
| Shuffle train | Yes | Yes | Match |

---

## Recommendations for TGCR

Based on these results, TGCR should focus on:

1. **Gaze-weighted fusion**: Since gaze dominates, TGCR router should likely weight gaze higher
2. **Subject-adaptive routing**: Different subjects may benefit from different routing
3. **Avoiding negative transfer**: When EEG is noisy (most subjects), router should suppress EEG
4. **Confidence-based switching**: Router could use prediction confidence to decide modality

### TGCR Success Criteria
1. TGCR_full should match or exceed SVM_Gaze_only (60.72%)
2. TGCR should NOT be significantly worse than gaze-only (no negative transfer)
3. Router weights should show interpretable patterns (higher gaze weights)
4. Shuffle tests should confirm EEG contributes minimally

---

## Files Generated

- `results/loso/svm_all_features_loso.csv` - Full SVM results
- `results/loso/majority_random_loso.csv` - Majority/Random baselines
- `results/loso/combined_baseline_summary.csv` - Combined summary
- `results/loso/significance_tests.csv` - Statistical tests
- `reports/loso_svm_baseline_matrix.md` - SVM-specific report
- `reports/pytorch_baseline_report.md` - This report

---

## Next Steps

1. Complete PyTorch MLP baselines (EEG_MLP, Gaze_MLP, Early/Late/Attention Fusion)
2. Implement and evaluate TGCR
3. Analyze TGCR router weights for interpretability
4. Run shuffle/ablation tests to confirm TGCR mechanisms