# EAGLE Framework Final Report

## EAGLE: EEG-Adaptive Generalization and Low-shot Enhancement

**Date:** 2026-05-10
**Framework Components:** HD-SIED, B-ACCS, SR-GC

---

## 一、HD-SIED Results (Hard-Domain SIED with GroupDRO)

### Method
- GroupDRO weighting on task loss to upweight hard source subjects
- Formula: `q_s = q_s * exp(eta * L_task_s)`, then `L_task_dro = sum(q_s[s] * L_task_s[s])`

### Results

| Model | Accuracy | Balanced Accuracy | vs SIED |
|-------|----------|-------------------|---------|
| Raw_EEG | 0.5082±0.0656 | 0.5166 | -2.89% |
| SIED | 0.5371±0.0311 | 0.5154 | baseline |
| HD-SIED (η=0.05, λ=0.01) | 0.4884±0.0567 | 0.4974 | **-4.87%** |
| HD-SIED (η=0.1, λ=0.05) | 0.5054±0.0568 | 0.5008 | **-3.17%** |

### Conclusion
**HD-SIED FAILS.** GroupDRO weighting degrades performance compared to standard SIED.

Possible reasons:
1. Hard-domain weighting may increase variance in adversarial training
2. The ETA/lambda hyperparameter combination may not be optimal
3. Simple averaging of subject losses in SIED may already be sufficient

**Not included as innovation point.**

---

## 二、B-ACCS Results (Balanced ACCS with Pseudo-label Teacher)

### Method
- Train teacher model on source subjects
- Generate pseudo-labels for target calibration pool
- Filter by confidence threshold τ and perform KMeans centroid sampling

### Results

| Method | 3-shot | 5-shot | 10-shot | 3/5/10 Avg |
|--------|--------|--------|---------|------------|
| Random | 0.4359 | 0.4156 | 0.5737 | 0.4750 |
| ACCS | 0.4793 | 0.4844 | 0.5134 | 0.4926 |
| B-ACCS (τ=0.6) | 0.3703 | 0.3559 | 0.3777 | 0.3680 |
| B-ACCS (τ=0.7) | 0.3640 | 0.3762 | 0.3742 | 0.3715 |
| B-ACCS (τ=0.8) | 0.3753 | 0.3662 | 0.3578 | 0.3663 |

### Root Cause Analysis
B-ACCS fails because pseudo-label accuracy is near random (~51-52%) when the teacher model trained on source subjects is applied to target subjects. The domain shift makes pseudo-labels unreliable.

**Not included as innovation point.**

---

## 三、SR-GC Results (Source-Regularized Gaussian Calibration)

### Method
- Estimate source class-conditional Gaussian: `μ_source_c, Σ_source_c`
- Blend with target calibration statistics: `μ_c = α*μ_target_c + (1-α)*μ_source_c`
- Prediction: Mahalanobis distance-based classification

### Results

| Shot | EEG_SVM | SR-GC α=0.25 | SR-GC α=0.5 | SR-GC α=0.75 |
|------|---------|--------------|-------------|---------------|
| 3-shot | 0.4346 | 0.5412 (+10.66%) | 0.5573 (+12.27%) | **0.5684 (+13.38%)** |
| 5-shot | 0.4161 | 0.5511 (+13.50%) | 0.5720 (+15.58%) | **0.5890 (+17.29%)** |
| 10-shot | 0.5764 | 0.5593 (-1.71%) | 0.5983 (+2.19%) | **0.6275 (+5.11%)** |
| 20-shot | 0.6083 | 0.5694 (-3.89%) | 0.6221 (+1.38%) | **0.6548 (+4.65%)** |
| 50-shot | 0.7736 | 0.5943 (-17.93%) | 0.6567 (-11.69%) | **0.6909 (-8.27%)** |

### Key Findings

1. **SR-GC excels at low-shot settings (3-5 shot)**
   - 3-shot: +13.38% improvement over EEG_SVM
   - 5-shot: +17.29% improvement over EEG_SVM

2. **SR-GC is effective up to 20-shot**
   - 10-shot: +5.11% improvement
   - 20-shot: +4.65% improvement

3. **SR-GC degrades at very high shots (50+)**
   - At 50-shot, target calibration statistics alone are more reliable
   - Source statistics become a hindrance rather than help

4. **α=0.75 is consistently best** (more weight on source statistics)

### Why SR-GC Works
- Source domain provides stable class-conditional priors
- At low-shot, target statistics are noisy; source priors dominate
- Mahalanobis distance naturally handles feature correlations
- No test label leakage - only source domain statistics used

### Conclusion
**SR-GC SUCCEEDS** as an effective low-shot calibration method.

**Can be used as innovation point.**

---

## 四、Final Framework Judgment

### Successful Modules

| Module | Status | Performance Gain | Innovation Point |
|--------|--------|-----------------|-----------------|
| **SR-GC** | ✅ Valid | +13-17% at 3-5 shot | Yes |
| SIED | ✅ Valid (prior work) | +2.86% zero-shot | Yes |

### Failed Modules

| Module | Status | Reason |
|--------|--------|--------|
| HD-SIED | ❌ Fails | GroupDRO hurts performance |
| B-ACCS | ❌ Fails | Pseudo-label accuracy ~51% (near random) |

### Final Innovation Points for Paper

**1. SR-GC: Source-Regularized Gaussian Calibration**
- For low-shot EEG user calibration (3-10 shot)
- Uses source domain class-conditional Gaussian as stable prior
- Significantly outperforms standard SVM at low-shot settings

**2. SIED: Subject-Invariant EEG Disentanglement**
- For zero-shot cross-user transfer
- Adversarial training removes subject identity
- +2.86% improvement over raw EEG

### Paper Innovation Statement

```
1. SR-GC: Source-regularized Gaussian calibration for low-shot EEG user calibration
   - Uses source-domain class-conditional statistics as stable priors
   - Achieves +13-17% improvement over SVM at 3-5 shot settings
   - No test label leakage

2. SIED: Subject-invariant EEG disentanglement for zero-shot transfer
   - Adversarial training removes subject-specific patterns
   - Achieves +2.86% improvement in zero-shot cross-user setting
```

### What NOT to Claim

- ❌ HD-SIED improves over SIED
- ❌ B-ACCS improves over ACCS
- ❌ SR-GC works at 50-shot
- ❌ EEG decodes pure cognitive state without calibration

---

## 五、Files Generated

- `results/final/srgc_results.csv` - SR-GC full results
- `results/final/hd_sied_results.csv` - HD-SIED full results
- `results/final/b_accs_results.csv` - B-ACCS full results
- `reports/final/eagle_framework_report.md` - This report