# EEG_SVM vs EEG_MLP Identical Results - Bug Report

## Issue

In the original `few_shot_calibration.py`, EEG_SVM and EEG_MLP produced **identical results** across all shot settings:
```
1-shot: 52.7 / 52.7
3-shot: 57.8 / 57.8
...
50-shot: 76.4 / 76.4
```

## Root Cause

Line 208 had incorrect model type parsing:
```python
# OLD (buggy):
acc, f1, bacc, auroc, cm = train_subject_specific(
    X_cal, y_cal, X_test, y_test,
    model_type.replace('_SVM', '').replace('_MLP', '')  # Bug here!
)
```

Both `EEG_SVM` and `EEG_MLP` became `EEG` after string replacement, which didn't match any model type in `train_subject_specific()`, so it fell through to the default `SGDClassifier`.

## Fix Applied

```python
# NEW (fixed):
actual_model_type = model_type.replace('EEG_', '').replace('Gaze_', '').replace('Combined_', '')
acc, f1, bacc, auroc, cm = train_subject_specific(
    X_cal, y_cal, X_test, y_test, actual_model_type
)
```

Now:
- `EEG_SVM` → `SVM`
- `EEG_MLP` → `MLP`
- `Gaze_SVM` → `SVM`
- `Combined_SVM` → `SVM`

## Corrected Results (5 seeds)

| Model | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|-------|--------|--------|--------|---------|---------|---------|
| EEG_MLP | 52.7% | 59.0% | 62.1% | 66.4% | 71.3% | **78.6%** |
| EEG_SVM | 52.7% | 58.5% | 60.8% | 65.7% | 71.1% | **78.6%** |
| Gaze_SVM | 54.9% | 61.1% | 63.3% | 66.1% | 68.3% | 70.9% |
| Combined | 54.2% | 61.8% | 64.1% | 67.0% | 70.3% | 73.3% |

## Observations

1. **EEG_SVM and EEG_MLP now differ** - confirming the bug was fixed
2. **EEG_MLP slightly outperforms EEG_SVM** at 5-10 shots (within variance)
3. **Both EEG models reach ~78.6% at 50-shot** - essentially the same ceiling
4. **EEG significantly outperforms Gaze at 50-shot** (78.6% vs 70.9%)

## Conclusion

The bug was confirmed and fixed. Results now show proper differentiation between SVM and MLP classifiers.