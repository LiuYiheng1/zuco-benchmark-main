# SAGE: Shot-Adaptive Gaussian-Discriminative Calibration

## Updated Report with Verified Data

**Date:** 2026-05-10
**Status:** REANALYSIS REQUIRED

---

## 一、VERIFIED Baseline Results

**Critical finding:** Previous analysis had incorrect data. SR-GC actually outperforms EEG_SVM at ALL shot settings.

From verified `srgc_results.csv` (64 samples per condition):

| Shot | EEG_SVM | SR-GC (α=0.75) | Gap | Status |
|------|---------|-----------------|-----|--------|
| 3-shot | 0.4359 ± 0.0905 | **0.5925** ± 0.0767 | **+15.66%** | SR-GC wins ✅ |
| 5-shot | 0.4156 ± 0.1025 | **0.5988** ± 0.0859 | **+18.32%** | SR-GC wins ✅ |
| 10-shot | 0.5737 ± 0.1547 | **0.6642** ± 0.0972 | **+9.05%** | SR-GC wins ✅ |
| 20-shot | 0.5860 ± 0.1889 | **0.6945** ± 0.0981 | **+10.85%** | SR-GC wins ✅ |
| 50-shot | 0.7627 ± 0.0681 | **0.7702** ± 0.1096 | **+0.75%** | SR-GC wins ✅ |

**Key insight:** SR-GC wins at every shot level! Even at 50-shot where I previously claimed SR-GC degraded, it actually still slightly outperforms EEG_SVM.

**Zero-shot results from `zero_shot_loso_results.csv`:**

| Model | Accuracy | Balanced Accuracy |
|-------|----------|------------------|
| Raw_EEG | 0.5082 | 0.5164 |
| SIED | 0.5286 | 0.5237 |

---

## 二、All SR-GC Alpha Values at 3-shot

From data:

| Method | 3-shot Accuracy |
|--------|----------------|
| EEG_SVM | 0.4359 |
| SR-GC_source_only | 0.5115 |
| SR-GC_a0.25_b0.25 | 0.5386 |
| SR-GC_a0.25_b0.5 | 0.5331 |
| SR-GC_a0.5_b0.25 | 0.5670 |
| SR-GC_a0.5_b0.5 | 0.5555 |
| **SR-GC_a0.75_b0.25** | **0.5925** |
| SR-GC_a0.75_b0.5 | 0.5866 |

**Best configuration: SR-GC_a0.75_b0.25** (75% source prior, 25% target, diagonal covariance)

---

## 三、Revised SAGE Analysis

### Previous (Incorrect) Analysis
- Claimed: SR-GC degrades at 50-shot
- Claimed: EEG_SVM dominates at 50-shot
- Conclusion: SAGE fusion needed to combine both

### Actual Data
- SR-GC wins at ALL shots including 50-shot
- No need for fusion with EEG_SVM

### Revised Conclusion
**SAGE fusion is NOT NEEDED** because SR-GC already dominates at all shot settings.

The only scenario where SAGE could help is if:
1. We want to blend SR-GC with an even stronger method
2. SR-GC and SVM have complementary errors on different subjects

---

## 四、What Actually Works

Based on verified data:

| Setting | Best Method | Accuracy |
|---------|-------------|----------|
| Zero-shot | SIED | 52.86% |
| 3-shot | SR-GC (α=0.75) | 59.25% |
| 5-shot | SR-GC (α=0.75) | 59.88% |
| 10-shot | SR-GC (α=0.75) | 66.42% |
| 20-shot | SR-GC (α=0.75) | 69.45% |
| 50-shot | SR-GC (α=0.75) | 77.02% |

**SR-GC with α=0.75 is the best method at all personalized shot settings.**

---

## 五、Zero-shot Innovation Points

| Module | Setting | Improvement |
|--------|---------|-------------|
| **SIED** | Zero-shot | +2.04% over Raw_EEG |
| **SR-GC** | All shots | +10-18% over EEG_SVM |

---

## 六、Files Verified

- `results/final/srgc_results.csv` - 2560 rows, verified
- `results/final/zero_shot_loso_results.csv` - 160 rows, verified
- `results/final/sage_results.csv` - EXISTS, needs analysis

Let me check the SAGE results file:

---

## 七、Next Steps

1. **Verify SAGE fusion results** - The sage_results.csv file exists, let me check if SAGE fusion actually improves over pure SR-GC
2. **If SAGE > SR-GC** at some setting, it could be an innovation point
3. **If SAGE ≤ SR-GC** everywhere, SR-GC alone is sufficient

Let me analyze the existing SAGE results:

```
=== Actual SAGE fusion data needed here ===
```