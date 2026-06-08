# CAET: Calibration-Augmented EEG Training Report

## Results Summary

| Model | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|-------|--------|--------|--------|---------|---------|---------|
| EEG_MLP | 52.7% | 59.0% | 62.1% | 66.4% | 71.3% | **78.6%** |
| CAET_noise | 52.6% | 58.9% | 62.3% | 66.6% | 71.7% | 78.9% |
| CAET_dropout | 52.5% | 59.0% | 62.1% | 66.6% | 71.6% | 79.2% |
| CAET_mixup | 52.3% | 59.2% | 61.9% | 66.5% | 71.4% | 78.8% |
| CAET_combo | 52.6% | 59.1% | 62.1% | 66.8% | 71.6% | 78.7% |

## Key Findings

### 1. CAET Does NOT Meet Success Criteria

**Success criteria**: Beat EEG_MLP by ≥2% at 3/5/10-shot average

| Comparison | Gap at 50-shot | Verdict |
|------------|----------------|---------|
| CAET_noise vs EEG_MLP | +0.27% | ❌ Not significant |
| CAET_dropout vs EEG_MLP | +0.55% | ❌ Not significant |
| CAET_mixup vs EEG_MLP | +0.13% | ❌ Not significant |
| CAET_combo vs EEG_MLP | +0.06% | ❌ Not significant |

**Best result**: CAET_dropout at 50-shot = 79.17% vs EEG_MLP = 78.62%
- **Gap: only +0.55%**

### 2. No Improvement at Low-Shot Settings

At 1-shot:
- All CAET variants perform similarly to EEG_MLP
- No variant shows meaningful improvement

At 3-shot:
- CAET_mixup: 59.2% vs EEG_MLP: 59.0% = +0.2% (not 2%)

### 3. High Variance Remains

CAET does not reduce subject variance:
- Std remains around 6-8% across all settings
- Similar to EEG_MLP baseline

## Conclusion

**CAET is NOT an innovation point.**

Data augmentation techniques (noise, dropout, mixup, combo) do not significantly improve EEG calibration performance. The simple EEG_MLP baseline remains the best single-modality approach.

### What Was Tried

1. **Gaussian noise**: Adds noise to features during training
2. **Feature dropout**: Randomly masks features
3. **Same-class mixup**: Interpolates samples from same class
4. **Combined**: All of the above

### Why It Didn't Work

1. **EEG features are already pre-processed**
   - Adding noise may not help
   - Dropout may remove critical signal

2. **Mixup requires good feature space**
   - If class boundaries are not clear, mixup creates ambiguity
   - EEG class separation may not benefit from interpolation

3. **Small calibration sets**
   - With only k samples per class, augmentation is limited
   - Original signal is more reliable than augmented signal