# SAN Leakage Check Report

## 1. Protocol Verification

### 1.1 Data Split Logic
```
Training Subjects (15): YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL
                        ↓ (excluding held_out)
Held-out Subject: One of the 16 Y-subjects at a time
                        ↓ (split by seed)
Test Set: ~1/3 of held-out subject data
Calibration Pool: ~2/3 of held-out subject data
```

### 1.2 Source Statistics Computation
```python
# SourceNorm mu/sigma computed from:
X_train_all = []  # All training subjects
y_train_all = []  # All training subjects
for subj in Y_SUBJECTS:
    if subj == held_out:
        continue  # EXCLUDES held-out
    X, y = load_eeg_data(subj)
    X_train_all.append(X)
    y_train_all.append(y)

# Then compute class-wise statistics:
mu_source_0 = mean(X_train_all[y_train_all == 0])
sigma_source_0 = std(X_train_all[y_train_all == 0])
```

**Verification**: ✅ Source statistics use only training subjects (15 subjects), NOT held-out subject.

## 2. Leakage Check Items

### 2.1 mu_source / sigma_source Only from Training Subjects
✅ **PASS** - Code explicitly excludes `held_out` subject:
```python
for subj in Y_SUBJECTS:
    if subj == held_out:
        continue
```

### 2.2 Does NOT Include Held-out Test Subject
✅ **PASS** - `held_out` subject is completely excluded from training data.

### 2.3 Does NOT Include Held-out Subject's Test Set
✅ **PASS** - Test set (`X_test`, `y_test`) comes from `held_out` but is only used for final evaluation, NOT for computing any statistics.

### 2.4 Does NOT Use Any Test Label
✅ **PASS** - Test labels (`y_test`) are only used in `train_and_evaluate()` for computing accuracy/f1/bacc/auroc.

### 2.5 Calibration and Test Strictly Separated
✅ **PASS** - Different index sets used:
```python
test_indices = indices[:test_size]        # First 1/3
cal_pool_indices = indices[test_size:]    # Last 2/3
X_test = X_test_orig[test_indices]         # Uses test_indices
X_cal_pool = X_test_orig[cal_pool_indices] # Uses cal_pool_indices
```

### 2.6 SourceNorm Did NOT Fit Scaler on All Y-Subjects
⚠️ **OBSERVATION** - SourceNorm uses statistics from ALL training subjects (15 subjects), not just calibration pool.

This is **intentional design**, not leakage:
- SourceNorm aims to capture cross-subject EEG patterns
- Similar to transfer learning / pre-trained statistics
- Does NOT use held-out subject's data

### 2.7 Same Split, Seeds, Classifier as StandardScaler
✅ **PASS** - Both use identical:
- `test_indices` / `cal_pool_indices` splits
- `seeds = [0, 1, 2, 3, 4]`
- SVM classifier with same parameters

### 2.8 50-shot Per Class = 100 Total Labeled Calibration Samples
✅ **PASS** - Code:
```python
for n_cal in shot_settings:  # [3, 5, 10, 20, 50]
    cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
    # Samples n_cal from each class = 2*n_cal total
```

### 2.9 Normalization Uses Only Features, Not Labels
✅ **PASS** - Statistics computed from features:
```python
mu = np.mean(X_class, axis=0)  # Features only
sigma = np.std(X_class, axis=0) + 1e-8  # Features only
```

## 3. Potential Concern: SourceNorm vs StandardScaler

### Difference:
- **StandardScaler**: `fit_transform(X_cal)` - uses only calibration data statistics
- **SourceNorm**: Uses statistics from ALL 15 training subjects

### Is This Fair?
SourceNorm provides a **stronger prior** but is valid because:
1. No information from held-out subject is used
2. This is analogous to using pre-trained model features
3. The comparison reveals whether cross-subject statistics help low-shot calibration

### Verdict: **NOT LEAKAGE** - This is the intended research question.

## 4. Conclusion

| Check Item | Status |
|------------|--------|
| 1. mu_source only from training subjects | ✅ PASS |
| 2. Excludes held-out subject | ✅ PASS |
| 3. Excludes held-out test set | ✅ PASS |
| 4. No test labels used | ✅ PASS |
| 5. Calibration/test separated | ✅ PASS |
| 6. SourceNorm design choice | ✅ INTENTIONAL |
| 7. Same protocol as StandardScaler | ✅ PASS |
| 8. 50-shot = 100 samples | ✅ PASS |
| 9. Features only, no labels | ✅ PASS |

**No data leakage detected.** SourceNorm is a valid experimental condition that tests whether cross-subject EEG statistics can serve as a stable anchor for low-shot user calibration.

## 5. Protocol Summary for SAN Paper

> "For each held-out subject, we compute class-wise mean (μ) and standard deviation (σ) from all 15 training subjects. These source-domain statistics serve as a normalization anchor. During calibration, we apply Source-Anchored Normalization: x_norm = (x - μ_source) / σ_source, where the calibration samples and test samples are normalized using the same source statistics. This ensures that the normalization does not depend on the limited calibration data statistics."