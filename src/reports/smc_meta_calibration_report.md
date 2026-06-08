# SMC Meta-Calibration Report

## Experiment Status

SMC (Subject-Meta Calibration) experiment is still running or encountered issues.

**Current Status**: Meta-learning experiments are computationally expensive and require significant training time.

---

## Preliminary Analysis

### Expected Benefits of Meta-Learning for EEG

Meta-learning approaches like Reptile and MAML are designed to:
1. Learn a good initialization for fast adaptation
2. Generalize across subjects/tasks
3. Improve few-shot performance

### Potential Issues

1. **EEG features are high-dimensional and noisy**
   - Meta-learning may not find a meaningful initialization
   - Subject variability is very high in EEG

2. **Small meta-training set**
   - Only 15 subjects for meta-training
   - Each subject has limited samples

3. **Meta-learning hyperparameters**
   - Inner/outer loop learning rates may need tuning
   - Number of adaptation steps is critical

---

## Alternative Interpretation

If SMC meta-learning doesn't improve over baseline, this would indicate:

1. **EEG features don't have a "fast adaptation" structure**
   - Each subject's EEG patterns are too unique
   - Meta-learning cannot leverage cross-subject patterns

2. **Simpler approaches work better**
   - Direct few-shot calibration (EEG_MLP baseline) is already effective
   - No need for complex meta-learning

---

## Recommendation

Wait for SMC experiment to complete. If results show:
- **SMC > EEG_MLP by ≥2% at 3/5/10-shot**: Claim as innovation
- **SMC ≈ EEG_MLP or worse**: Do NOT claim, use EEG_MLP baseline