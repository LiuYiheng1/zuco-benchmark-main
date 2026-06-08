# Reliability-Weighted Multi-Modal Calibration Report

## Experiment Summary

This experiment tested whether dynamically weighting EEG, Gaze, and Text-proxy predictions based on EEG calibration reliability could improve performance, especially on difficult subjects (YLS, YSL, YHS).

## Models Tested

| Model | Description |
|-------|-------------|
| EEG_only | EEG SVM classifier only |
| Gaze_only | Gaze SVM classifier only |
| Static_EEG_Gaze_average | 50/50 fusion of EEG and Gaze |
| Reliability_weighted_EEG_Gaze | Weight by EEG reliability: w_EEG = reliability, w_Gaze = 1-reliability |
| Reliability_weighted_EEG_TextProxy | Weight by EEG reliability: w_EEG = reliability, w_Text = 1-reliability |
| Reliability_weighted_EEG_Gaze_TextProxy | Three-way fusion weighted by reliability |

## Results: 50-shot Calibration

| Model | Accuracy | Macro F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|-------------------|-------|
| **Static_EEG_Gaze_average** | **82.62%** | **82.46%** | **82.68%** | **0.897** |
| Reliability_weighted_EEG_Gaze | 81.70% | 81.53% | 81.76% | 0.892 |
| Reliability_weighted_EEG_Gaze_TextProxy | 81.59% | 81.43% | 81.65% | 0.893 |
| Reliability_weighted_EEG_TextProxy | 81.47% | 81.29% | 81.51% | 0.893 |
| EEG_only | 78.78% | 78.59% | 78.81% | 0.858 |
| Gaze_only | 70.25% | 69.69% | 69.96% | 0.757 |

## Results: Calibration Curve

| Shot | EEG_only | Gaze_only | Static_Fusion | Rel_EEG+Gaze | Rel_EEG+Text | Rel_EEG+Gaze+Text |
|------|----------|-----------|---------------|--------------|--------------|-------------------|
| 1-shot | 47.3% | 45.1% | 46.0% | 46.0% | 45.8% | 46.0% |
| 3-shot | 51.8% | 57.2% | 57.3% | 57.3% | 57.5% | 57.4% |
| 5-shot | 55.4% | 59.6% | 60.7% | 60.1% | 59.6% | 59.9% |
| 10-shot | 62.0% | 64.5% | 67.2% | 66.6% | 67.3% | 66.8% |
| 20-shot | 67.9% | 65.8% | 72.0% | 71.6% | 71.8% | 71.7% |
| 50-shot | 78.8% | 70.2% | **82.6%** | 81.7% | 81.5% | 81.6% |

## Key Findings

### 1. Static Fusion Outperforms Reliability Weighting

**At 50-shot:**
- Static_EEG_Gaze_average: 82.62%
- Reliability_weighted_EEG_Gaze: 81.70%
- **Gap: -0.92%**

The simple 50/50 fusion outperforms dynamic reliability weighting.

### 2. All Fusion Methods Beat Single-Modality

- Static fusion (82.62%) beats EEG_only (78.78%) by **+3.84%**
- Even reliability weighting (81.59%) beats EEG_only by **+2.81%**

### 3. Gaze-Only is Weakest, But Helps in Fusion

Gaze-only achieves 70.25% at 50-shot, but when fused with EEG:
- Static fusion: 82.62% (vs EEG_only: 78.78%)
- This suggests Gaze provides complementary information to EEG

### 4. Difficult Subjects Performance

| Subject | EEG_only | Static_Fusion | Improvement |
|---------|----------|---------------|-------------|
| YLS | 72.6% | 76.3% | +3.7% |
| YSL | 72.5% | 78.4% | +5.9% |
| YHS | 76.6% | 80.9% | +4.3% |

Fusion improves performance on difficult subjects.

### 5. EEG Reliability Estimates

Average EEG reliability at different shot settings:
- 1-shot: 0.500 (no variance - single sample per class)
- 3-shot: 0.500 (same)
- 5-shot: 0.591
- 10-shot: 0.587
- 20-shot: 0.605
- 50-shot: 0.664

Higher reliability at 50-shot suggests EEG becomes more trustworthy with more calibration data.

## Analysis: Why Reliability Weighting Doesn't Beat Static Fusion

1. **EEG reliability is hard to estimate from small calibration sets**
   - At 1-3 shots, reliability cannot be properly estimated
   - Even at 50-shot, reliability is only 0.664

2. **Static fusion is already near-optimal for this dataset**
   - EEG and Gaze provide complementary information
   - A simple 50/50 split already captures most of the benefit

3. **Reliability estimation adds noise**
   - The CV-based reliability metric may not correlate with actual performance
   - Static fusion avoids this estimation error

## Conclusion: Static Fusion is the Best Innovation

### Innovation Points Ranked

1. **Static_EEG_Gaze_average** (82.62%): Best overall performance
2. **Reliability_weighted fusion** (81.59-81.70%): Slightly worse than static
3. **EEG_only** (78.78%): Good but worse than fusion
4. **Gaze_only** (70.25%): Weakest single modality

### What CAN Be Claimed as Innovation

1. **EEG-Gaze Static Fusion**: +3.84% over EEG_only, +12.37% over Gaze_only
2. **EEG reliability estimation**: Valid experimental finding showing EEG becomes more reliable with more calibration data
3. **Difficult subject improvement**: Fusion helps YLS, YSL, YHS by 3.7-5.9%

### What NOT to Claim

1. **Reliability weighting outperforms static fusion**: FALSE - static fusion is better
2. **User adapter improves over baseline**: FALSE - baseline MLP is best
3. **SIED-based methods are effective for within-subject**: FALSE - they all underperform

### Final Recommendations

**For the paper, claim:**
1. SIED for cross-subject generalization (validated: +3.55%)
2. EEG-Gaze static fusion for personalized prediction (validated: 82.62%)
3. Few-shot calibration protocol as experimental methodology

**Do NOT claim:**
1. TSPC, User Adapter, MCC (all failed ablation)
2. "EEG is strongest modality" (Gaze fusion is stronger)
3. "Adversarial training fully solves cross-user" (not fully solved)