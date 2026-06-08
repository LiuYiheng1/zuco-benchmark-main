# CLF: Calibrated Logit Fusion Report

## Results Summary

### 50-shot Calibration Performance

| Model | Accuracy | Macro F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|-----------------|-------|
| **Static_EEG_Gaze_average** | **82.62%** | **82.46%** | **82.68%** | **0.897** |
| CLF_temperature_scaled | 82.05% | 81.89% | 82.11% | 0.891 |
| CLF_logistic_stacking | 81.46% | 81.32% | 81.56% | 0.882 |
| Reliability_weighted_EEG_Gaze | 82.28% | 82.11% | 82.34% | 0.896 |
| EEG_only | 78.78% | 78.59% | 78.81% | 0.858 |
| Gaze_only | 70.25% | 69.69% | 69.96% | 0.757 |

### Calibration Curve

| Shot | EEG_only | Gaze_only | Static_Fusion | CLF_logistic | CLF_temp_scaled |
|------|----------|-----------|---------------|---------------|-----------------|
| 1-shot | 47.3% | 45.1% | 46.0% | 54.0% | 53.9% |
| 3-shot | 51.8% | 57.2% | 57.3% | 63.6% | 63.3% |
| 5-shot | 55.4% | 59.6% | 60.7% | 66.3% | 66.1% |
| 10-shot | 62.0% | 64.5% | 67.2% | 70.9% | 71.0% |
| 20-shot | 67.9% | 65.8% | 72.0% | 74.8% | 75.3% |
| 50-shot | 78.8% | 70.2% | **82.6%** | 81.5% | 82.1% |

## Key Findings

### 1. CLF Does NOT Beat Static Fusion at 50-shot

| Model | Accuracy | Gap vs Static |
|-------|----------|---------------|
| **Static_EEG_Gaze_average** | **82.62%** | - |
| CLF_temperature_scaled | 82.05% | -0.57% |
| CLF_logistic_stacking | 81.46% | -1.16% |
| Reliability_weighted | 82.28% | -0.34% |

### 2. CLF Does Beat Static Fusion at Lower Shots

At 20-shot:
- CLF_temperature_scaled: 75.3% vs Static: 72.0% = **+3.3%**

At 10-shot:
- CLF_temperature_scaled: 71.0% vs Static: 67.2% = **+3.8%**

### 3. CLF Shows Promise for Low-Shot Settings

The CLF approaches actually outperform static fusion at 1-20 shot settings!

- At 10-shot: CLF (70.9%) vs Static (67.2%) = +3.7%
- At 20-shot: CLF (75.3%) vs Static (72.0%) = +3.3%

### 4. Logistic Stacking vs Temperature Scaling

Both CLF methods perform similarly:
- CLF_logistic_stacking: 81.46%
- CLF_temperature_scaled: 82.05%

Temperature scaling is slightly better.

## Success Criteria Analysis

### CLF Success Criteria

> "CLF must exceed Static_EEG_Gaze_average by ≥1% at 50-shot, OR exceed by ≥1% at 20/50-shot average"

**50-shot result**: CLF (82.05%) < Static (82.62%) = **-0.57%** ❌

**20/50-shot average**:
- CLF: (75.3% + 82.1%) / 2 = 78.7%
- Static: (72.0% + 82.6%) / 2 = 77.3%
- **Gap: +1.4%** ✅

### Partial Success

CLF meets the success criteria for 20/50-shot average but not for 50-shot alone.

## Conclusion

**CLF is a partial success but may not be a strong innovation claim.**

### What Works
- CLF outperforms static fusion at 1-20 shot settings
- This is useful for low-shot scenarios

### What Doesn't Work
- At 50-shot, static fusion is still better
- The calibrated logit fusion adds complexity without clear benefit at high-shot

### Recommendation

**Do NOT claim CLF as a primary innovation.**

The reason: Static_EEG_Gaze_average (82.62%) remains the best at 50-shot, and the improvement at lower shots (+1.4% average) is not significant enough to claim as a novel method.

However, CLF can be mentioned as an **alternative approach** that performs comparably to static fusion.

## Final Innovation Points (Updated)

Based on all experiments:

1. **SIED** - Cross-subject generalization (+3.55%)
2. **Static_EEG_Gaze_fusion** - Best personalized prediction (82.62%)
3. **Few-shot calibration protocol** - Valid experimental methodology

**NOT validated as innovations:**
- TSPC, User Adapter, MCC, Reliability Weighting, CAET, CLF