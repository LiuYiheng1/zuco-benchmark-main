# ZuCo 2.0 EEG Usability Diagnostic Summary

## Date: 2026-05-08

## Executive Summary

**EEG signals ARE informative within subjects but FAIL completely at cross-subject generalization.**

| Experiment | Within-Subject | Cross-Subject (LOSO) | Gap |
|------------|----------------|----------------------|-----|
| EEG SVM | **86.00%** ± 6.93% | **50.60%** ± 5.87% | **-35.40%** |
| EEG MLP | **89.24%** ± 7.28% | ~52% (est) | **-37%** |

The **35% performance drop** from within-subject to cross-subject indicates EEG features encode strong subject-specific patterns (likely anatomical/procedural differences) that don't generalize across subjects.

---

## 1. Within-Subject EEG Classification Results

### Setup
- 16 Y-subjects
- 5-fold stratified CV per subject
- Models: SVM (linear) and MLP (2 hidden layers)
- Features: 420-dim EEG sentence-level features

### Overall Results

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|-------|----------|-----------|-------------------|
| **MLP** | **89.24%** ± 7.28% | 89.09% ± 7.41% | 89.24% ± 7.28% |
| SVM | 86.00% ± 6.93% | 85.75% ± 7.13% | 86.00% ± 6.93% |

### Per-Subject Results (SVM)

| Subject | Accuracy | Subject | Accuracy |
|---------|----------|---------|----------|
| YMD | 95.74% | YLS | 79.36% |
| YTL | 94.98% | YRP | 80.36% |
| YSD | 91.44% | YHS | 81.03% |
| YIS | 90.53% | YAG | 82.08% |
| YSL | 90.31% | YFS | 83.38% |
| YAK | 90.30% | YFR | 83.71% |
| YAC | 89.72% | YDG | 85.74% |
| YDR | 78.64% | YRK | 78.63% |

**Conclusion**: EEG has **very strong within-subject predictive power** (86-89%). This confirms EEG encodes task-relevant neural signals.

---

## 2. EEG PCA / Dimensionality Reduction Results

### Setup
- LOSO-Y protocol (16 folds)
- Methods: raw (420-dim), PCA 10/20/50/100 components, PCA 95% variance
- Classifier: Linear SVM (SGD)

### Results Summary

| Method | Accuracy | Std | Notes |
|--------|----------|-----|-------|
| pca_100 | 53.33% | ±5.16% | Best overall |
| pca_10 | 53.31% | ±4.20% | Nearly identical |
| pca_95pct | 51.93% | ±6.72% | ~40-44 components |
| pca_50 | 51.07% | ±4.80% | |
| raw | 50.60% | ±5.87% | Baseline |
| pca_20 | 49.36% | ±6.87% | Worst |

### Key Findings
1. **PCA does NOT help** - All methods cluster around 50%
2. **Higher dimensionality (100) slightly better** than raw (420)
3. **No consistent best method** - results highly variable across subjects

**Conclusion**: Dimensionality reduction alone cannot solve the cross-subject generalization problem.

---

## 3. EEG Feature Group Ablation Results

### Setup
- 420 features assumed to be 84 electrodes × 5 bands
- Tested contiguous groups of 84 features (bands 0-4)
- LOSO-Y protocol

### Results Summary

| Band (Features) | Accuracy | Std |
|-----------------|----------|-----|
| band_0_83 | 53.81% | ±4.54% |
| band_336_419 | 53.74% | ±6.74% |
| band_84_167 | 52.81% | ±4.46% |
| band_168_251 | 50.89% | ±6.78% |
| band_252_335 | 50.50% | ±6.26% |

### Key Findings
1. **No single frequency band is dominant** - all ~50-54%
2. **Subject variability dominates** - std is high (~5-7%)
3. **Band 0-83 (assumed theta/low) and 336-419 (assumed delta/high) slightly better**

**Important Caveat**: We don't have official documentation confirming the band structure. The ablation assumes contiguous feature groups correspond to frequency bands, but this may be incorrect.

**Conclusion**: Without knowing the exact feature structure, ablation is inconclusive. The problem is not about selecting the "right features" but about generalizing across subjects.

---

## 4. Subject Normalization / Alignment Results

### Setup
- LOSO-Y protocol
- Methods: global MinMaxScaler, train-only StandardScaler, subject-wise z-score
- Note: subject_zscore uses test subject's own statistics (unsupervised, no labels)

### Results Summary

| Method | Accuracy | Std |
|--------|----------|-----|
| global_minmax | 53.24% | ±5.55% |
| subject_zscore | 52.40% | ±7.57% |
| train_std | 50.60% | ±5.87% |

### Key Findings
1. **Normalization method has minimal impact** - all ~50-53%
2. **subject_zscore (using test subject's own stats) helps in some cases**:
   - YMD: 70.19% (vs 61.85% global_minmax) - **+8.34%**
   - YTL: 64.13% (vs 54.09% global_minmax) - **+10.04%**
   - YLS: 55.74% (vs 40.85% global_minmax) - **+14.89%**
3. **But hurts in other cases**:
   - YAC: 46.39% (vs 43.89%) - no improvement
   - YDG: 47.15% (vs 58.17%) - **-11.02%**

**Conclusion**: Subject-wise normalization shows promise but is highly subject-dependent. Some subjects benefit significantly, others are harmed.

---

## 5. Summary of All Experiments

| Experiment | Best Result | vs Random (50%) | vs Majority (54%) |
|------------|-------------|-----------------|-------------------|
| Within-Subject (MLP) | 89.24% | **+39.24%** | **+35.24%** |
| Within-Subject (SVM) | 86.00% | **+36.00%** | **+32.00%** |
| PCA (pca_100) | 53.33% | +3.33% | -0.67% |
| Feature Ablation (band_0_83) | 53.81% | +3.81% | -0.19% |
| Normalization (global_minmax) | 53.24% | +3.24% | -0.76% |
| Raw EEG LOSO | 50.60% | +0.60% | -3.40% |

---

## 6. Answers to Key Questions

### Q1: Is EEG completely useless, or is the extracted sentence-level EEG weak?

**EEG is NOT useless - it's STRONG within subjects but doesn't generalize.**

Within-subject accuracy of 86-89% proves EEG encodes task-relevant neural signals. The problem is **cross-subject generalization**, not signal quality.

### Q2: Does EEG have within-subject information?

**YES, definitively.** Both SVM (86%) and MLP (89%) achieve high within-subject accuracy. The EEG features do discriminate between NR and TSR within a single subject.

### Q3: Is EEG mainly affected by cross-subject differences?

**YES.** The 35% gap between within-subject and cross-subject performance indicates:
1. **Subject-specific artifacts** (headcap placement, impedance, noise)
2. **Anatomical differences** (brain morphology, electrode positioning)
3. **Procedural differences** (time of day, fatigue, attention)

### Q4: Can PCA / feature selection improve EEG?

**NO.** PCA at various dimensions (10, 20, 50, 100, 95% variance) all produce ~50-53% accuracy, essentially the same as raw features. Feature selection alone cannot fix the generalization problem.

### Q5: Is it worth continuing with EEG as the primary model?

**NOT for cross-subject classification with current features.**

Given:
- Within-subject: 86-89%
- Cross-subject: ~50% (near random)
- No improvement from PCA, normalization, or feature selection

The current EEG features are **not suitable for cross-subject classification** without significant advancement in domain adaptation or subject alignment.

### Q6: If worth pursuing, should we focus on word/fixation-level EEG or subject alignment?

**Subject alignment is the more promising direction** based on the data:

1. Within-subject works → subject-specific patterns exist
2. Subject-wise z-score helped some subjects significantly (YLS: +15%, YMD: +8%, YTL: +10%)
3. The problem is domain shift between subjects

**Possible next steps**:
- CORAL (Correlation Alignment) for domain adaptation
- Subject-adaptive normalization (using test subject's unlabeled data)
- Adversarial domain adaptation
- Simple subject embedding / calibration

### Q7: Should the main task shift to gaze-based reading behavior modeling?

**YES - and this is already supported by the data.**

Gaze-only LOSO accuracy: **60.72%** (significantly above EEG's 50.60%)

Gaze advantages:
1. **Higher cross-subject accuracy** (60.72% vs 50.60%)
2. **Lower feature dimensionality** (9 vs 420) → less overfitting
3. **Behavioral measure** - less susceptible to anatomical differences
4. **Task-relevant** - eye movements directly reflect reading strategy

---

## 7. Recommendations

### Immediate Actions

1. **Shift primary modeling to gaze-based approach**
   - SVM_Gaze_only (60.72%) is the strongest baseline
   - Focus on improving gaze features or gaze-specific models

2. **Do NOT invest more in EEG-only cross-subject models**
   - Current features are insufficient for cross-subject generalization
   - Would require significant feature engineering or domain adaptation

3. **If continuing EEG research**, prioritize:
   - Subject alignment techniques (CORAL, AdaIN, etc.)
   - Unsupervised test-time adaptation
   - Riemannian manifold approaches for EEG

### Long-term Considerations

1. **Collect subject-calibrated EEG features** if EEG is critical
2. **Consider within-subject experimental design** if cross-subject is not required
3. **Explore word-level or fixation-level EEG** for finer-grained analysis

---

## 8. Files Generated

| File | Description |
|------|-------------|
| `results/eeg_diagnostics/within_subject_eeg.csv` | Per-fold within-subject results |
| `results/eeg_diagnostics/within_subject_eeg_summary.csv` | Summary statistics |
| `results/eeg_diagnostics/eeg_pca_loso.csv` | PCA experiment results |
| `results/eeg_diagnostics/eeg_feature_group_ablation.csv` | Feature group ablation results |
| `results/eeg_diagnostics/eeg_alignment_loso.csv` | Normalization experiment results |
| `reports/eeg_usability_diagnostic_summary.md` | This report |

---

## 9. Conclusion

**The EEG signals are strong within subjects (86-89%) but completely fail at cross-subject generalization (~50%). This 35% gap is likely due to subject-specific anatomical and procedural differences that are not captured by the current sentence-level EEG features.**

**The recommended path forward is to focus on gaze-based modeling (60.72% cross-subject accuracy) rather than continuing to invest in EEG-only approaches without significant domain adaptation.**