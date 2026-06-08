# GACS: Gaussian-prior Active Calibration Sampling

## Final Report

**Date:** 2026-05-10
**Status:** FAILED - Does not meet success criteria

---

## 一、GACS Results (3 seeds, 16 subjects)

| Shot | Random | GACS_anchor | GACS_boundary | GACS_balanced |
|------|--------|--------------|---------------|---------------|
| 1-shot | 0.5084 | **0.5238** | **0.5286** | - |
| 3-shot | **0.5851** | 0.5482 | 0.5703 | 0.5568 |
| 5-shot | **0.6156** | 0.5609 | 0.5949 | 0.5865 |
| 10-shot | **0.6563** | 0.6197 | 0.6150 | 0.6110 |
| 20-shot | **0.7078** | 0.6703 | 0.6549 | 0.6416 |
| 50-shot | **0.7858** | 0.7726 | 0.7564 | 0.7184 |

### Gap vs Random

| Shot | GACS_anchor | GACS_boundary | GACS_balanced |
|------|-------------|---------------|---------------|
| 1-shot | **+1.54%** | **+2.02%** | - |
| 3-shot | -3.69% | -1.48% | -2.83% |
| 5-shot | -5.47% | -2.07% | -2.91% |
| 10-shot | -3.66% | -4.13% | -4.53% |
| 20-shot | -3.75% | -5.29% | -6.62% |
| 50-shot | -1.32% | -2.94% | -6.74% |

---

## 二、Success Criteria Check

| Criterion | Result |
|-----------|--------|
| 1-shot not worse than Random | ✅ GACS_boundary wins |
| 3-shot not worse than Random | ❌ All GACS below Random |
| 5/10-shot exceeds Random | ❌ All GACS below Random |
| Average exceeds Random and KMeans | ❌ Fails |

**GACS Status: FAILED**

---

## 三、Root Cause Analysis

### Why GACS Fails

1. **Pseudo-label quality issue**:
   - Source-trained Gaussian may not transfer well to target subjects
   - Pseudo-labels from source prior may be incorrect for target distribution

2. **Sampling bias**:
   - High-confidence samples from source Gaussian may not represent target distribution
   - Boundary sampling focuses on decision boundary but may select mislabeled samples

3. **Diversity loss**:
   - GACS selects based on pseudo-labels, not true labels
   - May over-sample one class while missing the other

### Key Insight

**GACS only wins at 1-shot** (where Random is essentially random), but **loses at all other shots**.

This suggests that:
- Source Gaussian prior is useful only when we have almost no data (1-shot)
- At higher shots, Random sampling is more representative
- Active sampling with noisy pseudo-labels hurts more than helps

---

## 四、Conclusion

**GACS is NOT a valid innovation point** because:

1. It loses to Random at 3+ shot
2. The source Gaussian prior introduces bias rather than helping
3. No scenario where GACS significantly outperforms Random

### True Best Methods

| Setting | Best Method |
|---------|------------|
| Zero-shot | SIED |
| Low-shot (1-3) | Random or GACS at 1-shot only |
| Medium-shot (5-20) | Random |
| High-shot (50+) | EEG_SVM |

**Final Innovation Points remain:**
1. **SIED** - Zero-shot transfer
2. **SR-GC** - Low-shot Gaussian calibration (but note: this was earlier, needs verification)