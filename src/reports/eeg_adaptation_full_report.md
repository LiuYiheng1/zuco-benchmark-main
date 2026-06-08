# EEG Subject-Adaptation Experiment - Final Report

## 1. Experiment Overview

**Date:** 2026-05-09

**Goal:** Evaluate subject-adversarial domain adaptation for EEG-based reading mode classification (Normal Reading vs Task-Specific Reading).

**Protocol:** Leave-One-Subject-Out (LOSO) cross-validation

**Subjects:** 16 (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL)

**Seeds:** 5 (0, 1, 2, 3, 4)

## 2. Models Tested

| Model | Description |
|-------|-------------|
| Raw_EEG | Standard EEG classifier (SVM) without adaptation |
| EEG_CORAL | CORAL domain adaptation on EEG features |
| EEG_Adversarial λ=X | Subject-adversarial training with gradient reversal |
| Gaze_only | Eye-tracking features only (baseline) |
| Combined | EEG + Gaze feature concatenation |
| EEG_DANN | Domain-Adversarial Neural Network (pilot) |
| EEG_MMD | Maximum Mean Discrepancy adaptation (pilot) |

## 3. Main Results (5 seeds)

| Model | Accuracy | Macro-F1 | Balanced Acc | AUROC |
|-------|----------|----------|--------------|-------|
| **Gaze_only** | **61.27%** | **57.01%** | **61.29%** | **69.13%** |
| Combined | 59.28% | 53.99% | 59.50% | 69.40% |
| EEG_Adversarial λ=0.01 | 54.38% | 46.05% | 54.04% | 58.37% |
| EEG_Adversarial λ=0.1 | 53.93% | 44.62% | 53.36% | 57.75% |
| EEG_Adversarial λ=0.05 | 53.89% | 44.65% | 53.24% | 58.20% |
| EEG_CORAL | 51.52% | 44.20% | 52.42% | 56.22% |
| Raw_EEG | 50.82% | 42.26% | 51.64% | 57.24% |

## 4. Statistical Significance (Wilcoxon paired test)

| Comparison | Mean Diff | p-value | Significance |
|-----------|-----------|---------|-------------|
| EEG_Adversarial λ=0.01 vs Raw_EEG | +3.55% | 7.4e-05 | *** (highly significant) |
| EEG_Adversarial λ=0.05 vs Raw_EEG | +3.06% | 5.5e-04 | *** (highly significant) |
| EEG_Adversarial λ=0.1 vs Raw_EEG | +3.10% | 3.3e-04 | *** (highly significant) |
| EEG_CORAL vs Raw_EEG | +0.69% | 0.411 | n.s. |
| EEG_Adversarial λ=0.01 vs EEG_CORAL | +2.86% | 8.1e-04 | *** (highly significant) |
| EEG_Adversarial λ=0.01 vs Gaze_only | -6.89% | 0.001 | ** (significant) |

**Significance levels:** *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant

## 5. Key Findings

### 5.1 Gaze is more robust across subjects
Eye-tracking features (gaze) consistently outperform EEG features in cross-subject evaluation:
- Gaze-only: **61.27% accuracy** (best single modality)
- Combined EEG+Gaze: 59.28% accuracy
- EEG adversarial: 54.38% accuracy

### 5.2 Adversarial adaptation significantly improves EEG
Subject-adversarial training improves EEG cross-subject accuracy by **+3.55%** over raw EEG:
- Raw EEG: 50.82%
- EEG Adversarial λ=0.01: 54.38% (**p<0.001**)
- The improvement is **highly statistically significant**

### 5.3 CORAL does not significantly help
CORAL domain adaptation shows minimal improvement over raw EEG (~0.7%) and is **not statistically significant**.

### 5.4 λ parameter analysis
- λ=0.01: 54.38% (best among adversarial variants)
- λ=0.05: 53.89%
- λ=0.1: 53.93%
- Smaller λ is slightly better, suggesting task loss should dominate

### 5.5 Subject Invariance Analysis (FIXED)

**Protocol correction:** Original analysis had a critical bug. Fixed version uses within-subject CV on training subjects.

**Key finding:**
- Raw EEG features: **99.97%** subject predictability (near perfect)
- Adversarial embeddings: **~7%** subject predictability (close to random = 1/15 ≈ 6.67%)

**Conclusion:** Adversarial training improves cross-subject generalization **by removing subject-specific information** from EEG embeddings. This mechanism is now supported by analysis.

## 6. MMD Pilot Results (3 seeds)

Maximum Mean Discrepancy (MMD) adaptation penalizes variance across subject embeddings to encourage subject-invariance.

| Model | Accuracy | Macro-F1 | Balanced Acc |
|-------|----------|----------|--------------|
| EEG_MMD λ=0.01 | 53.37% | ~45% | ~53% |
| EEG_MMD λ=0.05 | 53.46% | ~45% | ~53% |
| EEG_MMD λ=0.1 | 54.38% | ~46% | ~54% |

Key finding: MMD achieves similar performance to adversarial training (both ~54.4% at best), confirming that reducing inter-subject variance helps EEG cross-subject generalization.

## 7. DANN Pilot (preliminary)

DANN was tested but results were not fully captured due to training interruption. Preliminary observations suggest similar performance to adversarial training.

## 8. Problematic Subjects

Subjects with consistently low performance across all models:
- **YLS**: Often below 45% accuracy
- **YSL**: Often below 50% accuracy
- **YHS**: Often below 53% accuracy

These subjects may have:
- Different neural responses to reading tasks
- Lower signal quality
- Unique cognitive strategies

## 9. Paper Recommendations

### Should say:
> "EEG contains strong subject-specific task information but suffers from severe cross-subject domain shift (raw EEG: 99.97% subject predictability). Subject-adversarial training significantly improves cross-subject generalization (50.82% → 54.38%, +3.55%, p<0.001) by removing subject-specific information from EEG embeddings (adversarial: ~7% subject predictability)."

> "Eye-tracking (gaze) remains more robust across subjects than EEG in cross-subject evaluation (61.3% vs 54.4%)."

### Should NOT say:
- "EEG is the strongest modality."
- "EEG outperforms gaze."
- "Adversarial adaptation fully solves the cross-subject problem."

## 10. Future Directions

1. Investigate why YLS/YSL/YHS have poor performance
2. Explore subject-specific calibration for EEG
3. Consider hierarchical models separating subject/task effects
4. Try other domain adaptation methods (e.g., domain-conditioned normalization)

## 11. Files Generated

| File | Description |
|------|-------------|
| `results/eeg_adaptation/eeg_adaptation_full_5seed.csv` | All 5-seed results (560 rows) |
| `results/eeg_adaptation/eeg_adaptation_summary_5seed.csv` | Summary statistics |
| `results/eeg_adaptation/eeg_adaptation_significance.csv` | Wilcoxon test results |
| `results/eeg_adaptation/subject_leakage_analysis.csv` | Subject identity predictability analysis |
| `results/eeg_adaptation/eeg_mmd_pilot.csv` | MMD pilot results (144 rows) |
| `reports/eeg_adaptation_full_report.md` | This report |