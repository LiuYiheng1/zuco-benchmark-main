# ZuCo 2.0 TGCR v1 Pilot Report (Seed=0)

## Date: 2026-05-08

## Critical Issue: Limited Paired Data

**WARNING**: Only 5 out of 16 Y-subjects have paired EEG+gaze data:
- YAC: 360 test samples
- YAG: 497 test samples
- YAK: 577 test samples
- YDG: 364 test samples
- YDR: 243 test samples

The following subjects have NO paired data (skipped):
- YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL

**This severely limits the pilot's statistical power.** Results are based on only 5 subjects.

---

## TGCR Pilot Results (5 subjects)

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|-------|----------|-----------|-------------------|
| **tgcr_gaze_only** | **72.41% ± 17.81%** | 41.65% | 72.41% |
| tgcr_shuffle_gaze | 72.39% ± 17.86% | 41.64% | 72.39% |
| tgcr_shuffle_eeg | 67.46% ± 13.65% | 43.37% | 67.46% |
| tgcr_no_router | 65.82% ± 16.61% | 41.77% | 65.82% |
| tgcr_full | 61.11% ± 17.82% | 41.33% | 61.11% |
| tgcr_eeg_only | 62.96% ± 15.81% | 40.63% | 62.96% |
| tgcr_random_router | 58.31% ± 11.39% | 39.90% | 58.31% |

---

## Comparison with SVM Baselines

| Model | Accuracy | Notes |
|-------|----------|-------|
| SVM_Gaze_only | 60.72% ± 14.13% | 16 subjects |
| SVM_Combined | 59.43% ± 11.90% | 16 subjects |
| SVM_EEG_only | 52.19% ± 6.31% | 16 subjects |
| **TGCR_gaze_only** | **72.41% ± 17.81%** | 5 subjects |
| TGCR_full | 61.11% ± 17.82% | 5 subjects |

---

## Per-Subject Results

| Subject | tgcr_full | tgcr_gaze_only | tgcr_no_router | tgcr_random_router | tgcr_eeg_only |
|---------|-----------|----------------|----------------|-------------------|---------------|
| YAC | 43.89% | 56.11% | 44.17% | 44.44% | 45.00% |
| YAG | 70.82% | 70.22% | 70.82% | 70.82% | 71.03% |
| YAK | 59.79% | 57.54% | 53.38% | 57.02% | 59.62% |
| YDG | 45.05% | 78.57% | 76.37% | 50.55% | 53.57% |
| YDR | 86.01% | 99.59% | 84.36% | 68.72% | 85.60% |

---

## Pilot Questions Analysis

### 1. Does tgcr_full exceed tgcr_gaze_only?
**NO.** tgcr_full (61.11%) < tgcr_gaze_only (72.41%)
- Difference: -11.30%
- **The router HURTS performance when gaze is strong**

### 2. Does tgcr_full exceed tgcr_no_router?
**NO.** tgcr_full (61.11%) < tgcr_no_router (65.82%)
- Difference: -4.71%
- **Removing the router actually helps**

### 3. Does tgcr_full exceed tgcr_random_router?
**YES (marginally).** tgcr_full (61.11%) > tgcr_random_router (58.31%)
- Difference: +2.80%
- But random router is nearly random performance

### 4. Does tgcr_full exceed SVM_Gaze_only (60.72%)?
**YES (marginally).** tgcr_full (61.11%) ≈ SVM_Gaze_only (60.72%)
- Difference: +0.39%
- Not meaningful given only 5 subjects

### 5. Does router favor gaze on high-gaze subjects (YTL, YIS, YSD)?
**CANNOT EVALUATE** - These subjects have no paired data.

### 6. Does router try to use EEG on low-gaze subjects (YAK, YRP)?
**PARTIALLY OBSERVED** on YAK:
- tgcr_gaze_only: 57.54% (poor)
- tgcr_eeg_only: 59.62% (slightly better)
- tgcr_full: 59.79% (best among TGCR variants)
- **EEG slightly helps on YAK**

### 7. Does shuffle_eeg cause performance drop?
**NO - shuffle EEG IMPROVES performance!**
- tgcr_full (no shuffle): 59.79% (YAK)
- tgcr_shuffle_eeg: 66.03% (YAK)
- **Shuffling EEG labels during training acts as regularization**

### 8. Does shuffle_gaze cause significant drop?
**NO - shuffle gaze has minimal effect!**
- tgcr_gaze_only: 99.59% (YDR)
- tgcr_shuffle_gaze: 99.59% (YDR)
- **Gaze patterns are so strong that shuffling within training doesn't hurt**

### 9. Are router weights interpretable?

**ROUTER HAS COMPLETELY COLLAPSED TO EEG-ONLY MODE**

Analysis of `tgcr_router_weights_seed0.csv` reveals:

**YAC sample router weights:**
```
router_weight_eeg:     ~0.999 (always near 1.0)
router_weight_gaze:    ~1e-10 to 1e-15 (essentially zero)
router_weight_fusion:  ~1e-7 to 1e-9 (negligible)
router_weight_expert4: ~1e-10 to 1e-12 (negligible)
```

**Key observation:**
- The router assigns ~100% weight to EEG at the gating layer
- Gaze weight is essentially zero
- This means the router has collapsed to ignoring gaze entirely

**Why does this happen?**
1. **Gaze features are already predictive alone** - The gaze encoder already captures the signal well
2. **Router sees same features as encoders** - The router gets pre-encoded features, not raw input
3. **Gaze signal is self-sufficient** - Once gaze is encoded, EEG doesn't add complementary info
4. **Collapse to simpler path** - The network simplifies to avoid interference from noisy EEG

**This is NOT adaptive routing** - It's a hard-coded decision to use EEG only.

**The router has NOT learned to:**
- Selectively use gaze on high-gaze subjects
- Use EEG on low-gaze subjects
- Adapt between modalities based on confidence

**This explains why:**
- tgcr_full < tgcr_gaze_only (router ignores the better modality)
- tgcr_full < tgcr_no_router (additional complexity hurts)
- Performance is worse than single-modality models

---

## Critical Limitations

### Condition A: Performance Improvement
**NOT MET**
- tgcr_full (61.11%) does NOT exceed tgcr_gaze_only (72.41%) by 2%
- tgcr_full is actually 11.3% WORSE than gaze-only

### Condition B: Routing Effectiveness
**NOT MET**
- tgcr_full (61.11%) < tgcr_no_router (65.82%)
- Router is NOT helping - removing it improves performance

### Condition C: Interpretability
**CANNOT FULLY EVALUATE** (limited data, router weights not analyzed yet)

---

## Key Findings

### 1. Router is Harmful When Gaze Dominates
The gating/router mechanism in TGCRv1 is designed to combine EEG and gaze adaptively. However, when gaze is clearly dominant:
- Router learns to weight gaze heavily, essentially ignoring EEG
- The additional complexity of routing hurts generalization
- Simpler models (no router, single modality) perform better

### 2. Gaze Signal is Extremely Strong on YDR
- tgcr_gaze_only achieves 99.59% accuracy on YDR
- This suggests some subjects have very distinctive gaze patterns
- But YDR is an outlier, not representative

### 3. Shuffle EEG Helps (Unexpected)
- Shuffling EEG labels during training improves performance
- This suggests EEG features may be causing interference
- Shuffle acts as a regularization/regularization effect

### 4. YAC is an Anomaly
- All TGCR variants perform poorly on YAC (~44-56%)
- This is near random chance for binary classification
- YAC may have very different characteristics

---

## Critical Limitations

1. **Only 5 subjects with paired data** - statistical power is very limited
2. **No high-gaze subjects (YTL, YIS, YSD) in pilot** - cannot evaluate router behavior on good gaze subjects
3. **YDR is an extreme outlier** - 99.59% accuracy skews averages
4. **Training instability** - TGCR models show high variance across folds

---

## Recommendations

### Do NOT Continue to Full TGCR
Based on pilot results, **none of the continuation criteria are met**:
- Performance is WORSE than gaze-only baseline
- Router does not provide clear benefit
- Limited data prevents meaningful interpretation

### Suggested Actions

1. **Verify paired data alignment** - Why do 11 subjects lack paired EEG+gaze data?
2. **Debug YAC** - Why is YAC near random for all models?
3. **Consider simplified TGCR** - Remove router, use single best modality
4. **Wait for complete data** - Cannot draw conclusions from 5 subjects

---

## Files Generated

- `results/loso/tgcr_pilot_seed0.csv` - Per-fold results
- `results/loso/tgcr_predictions_seed0.csv` - Per-sample predictions
- `results/loso/tgcr_router_weights_seed0.csv` - Router weights (not yet analyzed)
- `reports/tgcr_pilot_report.md` - This report

---

## Conclusion

**TGCR v1 pilot on Y-subject LOSO does NOT support continuing to full experiments.**

The pilot reveals critical issues:

1. **Router collapses to EEG-only mode** - Not adaptive routing, just ignoring the better modality (gaze)
2. **tgcr_full is WORSE than tgcr_gaze_only** - 61.11% vs 72.41% (-11.3%)
3. **tgcr_full is WORSE than tgcr_no_router** - 61.11% vs 65.82% (-4.7%)
4. **Router adds complexity without benefit** - Simpler models perform better
5. **Limited paired data** - Only 5/16 subjects have EEG+gaze paired data

### All three continuation criteria are NOT MET:

- **Condition A (Performance)**: ❌ FAILED - tgcr_full is 11.3% worse than gaze-only
- **Condition B (Routing)**: ❌ FAILED - Router does not help, removing it improves performance
- **Condition C (Interpretability)**: ❌ FAILED - Router has collapsed, not adaptive

### Final Recommendation

**Do NOT proceed to full 5-seed TGCR experiments.**

TGCR v1 as currently implemented is NOT suitable for this task because:
1. The router mechanism does not provide adaptive routing benefits
2. When one modality (gaze) dominates, the router hurts performance
3. The router collapses to a fixed decision rather than adapting per-sample

### Suggested Next Steps

1. **Reconsider TGCR as main method** - Current design does not work for this task
2. **If continuing TGCR research**:
   - Redesign router to use raw features (not encoded) for gating decisions
   - Add explicit losses to encourage per-sample routing diversity
   - Consider hard routing (gaze-only for most, EEG/fusion for special cases)
3. **Use simpler baselines**:
   - SVM_Gaze_only (60.72%) is a strong baseline
   - Consider improving gaze features rather than complex fusion