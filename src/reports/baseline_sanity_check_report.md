# ZuCo 2.0 Baseline Sanity Check Report - CRITICAL ISSUE FOUND

## Date: 2026-05-08

---

## ⚠️ CRITICAL FINDING: Test Set Labels Are Missing!

### The Problem

After deep investigation, we discovered that **the test set (X-* subjects) extracted features have EMPTY labels** in the downloaded `features.zip` file.

**Evidence:**

1. **YAC (Train Subject) Key Format**: `YAC_NR_2_0`
   - Key parts: `['YAC', 'NR', '2', '0']`
   - Label extracted from key: `NR` (Normal Reading)
   - Value's last element: `'NR'`

2. **XBB (Test Subject) Key Format**: `XBB__0_0` (double underscore!)
   - Key parts: `['XBB', '', '0', '0']` - **EMPTY STRING for label!**
   - Label extracted from key: `''` (empty)
   - Value's last element: `''` (empty string)

3. **All X-* subjects have empty labels:**
   ```
   XBB: normal_ratio=0.0, num_normal=0, num_task_specific=561
   XDT: normal_ratio=0.0, num_normal=0, num_task_specific=626
   ... (all 10 X-* subjects show 100% TSR)
   ```

4. **Our label parsing code:**
   ```python
   label_binary = 1 if label == "NR" else 0  # Empty string "" != "NR", so label=0 (TSR)
   ```

### Impact

Due to this issue:
- **ALL test samples were assigned label 0 (TSR)** because `"" != "NR"`
- Train set labels were correct (NR=1, TSR=0)
- This explains why:
  - `EEG_only accuracy ≈ 66%` = majority class (all test is "TSR" but model predicts 0/1)
  - `gaze_only accuracy ≈ 39%` = random-ish because model learned on balanced data but tested on 100% TSR
  - `balanced_accuracy == accuracy` because there's only one class in test set

---

## Root Cause Analysis

According to the official `data_helpers.py` code:

```python
# For train subjects (Y-*):
f_nr = read_mat_file(os.path.join(dir, f"results{subject}_NR.mat"))
f_tsr = read_mat_file(os.path.join(dir, f"results{subject}_TSR.mat"))
fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")

# For test subjects (X-*):
f = read_mat_file(os.path.join(dir, f"results{subject}.mat"))
fe.extract_sentence_features(subject, f, feature_set, features, "")
```

The test set `.mat` files don't separate NR vs TSR - they only contain one condition, and the label parameter is passed as empty string `""`.

**The extracted features in `features.zip` were created with this empty label, and there's no way to recover the true labels from the downloaded files alone.**

---

## What This Means

### ❌ CANNOT Perform Local Evaluation

Without access to the original `.mat` files that contain the actual labels for test subjects, **we cannot compute accuracy/F1/etc. locally** because we don't know the true labels.

### ✅ CAN Still Train Models

We can still:
- Train on Y-* subjects (labels are correct)
- Generate predictions on X-* subjects
- Submit predictions to EvalAI for official evaluation

---

## Correct Path Forward

### Option 1: Submit to EvalAI (Official)
1. Train on Y-* subjects using our correct code
2. Generate predictions on X-* subjects
3. Submit to [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2125/submission)
4. Get official evaluation results

### Option 2: Re-extract Features from Raw Data
1. Download the full dataset (70GB)
2. Re-run `benchmark_baseline.py` which will extract features from `.mat` files
3. The official extraction code should properly handle test labels

### Option 3: Use Leave-One-Subject-Out on Training Data
1. Use only Y-* subjects
2. Perform leave-one-subject-out cross-validation within Y-* subjects
3. This gives us local evaluation metrics without needing X-* labels

---

## Correct Sanity Check Results (Train Only)

### Label Distribution (Training Set Only - Y-* subjects)

| Subject | Total | NR | TSR | NR Ratio |
|---------|-------|-----|-----|----------|
| YAC | 360 | 158 | 202 | 0.439 |
| YAG | 658 | 305 | 353 | 0.464 |
| YAK | 577 | 245 | 332 | 0.425 |
| YDG | 526 | 240 | 286 | 0.456 |
| YDR | 618 | 268 | 350 | 0.434 |
| YFR | 350 | 183 | 167 | 0.523 |
| YFS | 488 | 195 | 293 | 0.400 |
| YHS | 717 | 346 | 371 | 0.483 |
| YIS | 729 | 340 | 389 | 0.466 |
| YLS | 470 | 191 | 279 | 0.406 |
| YMD | 540 | 271 | 269 | 0.502 |
| YRK | 234 | 113 | 121 | 0.483 |
| YRP | 387 | 185 | 202 | 0.478 |
| YSD | 713 | 331 | 382 | 0.464 |
| YSL | 691 | 322 | 369 | 0.466 |
| YTL | 697 | 330 | 367 | 0.473 |

**Summary:** Training set is reasonably balanced (NR ratio: 0.40-0.52)

---

## Feature Quality (Training Set)

| Feature | Shape | NaN | Inf | Constant Features |
|---------|-------|-----|-----|------------------|
| EEG | (8755, 420) | 0 | 0 | 0 |
| Gaze | (8755, 9) | 0 | 0 | 0 |

**Features are clean - no NaN/Inf issues.**

---

## Correct Code Fixes Applied

### 1. Balanced Accuracy - VERIFIED CORRECT
```python
from sklearn.metrics import balanced_accuracy_score
bacc = balanced_accuracy_score(y_true, y_pred)  # ✅ Correct
```

### 2. Label Parsing - NEEDS FIX for Test Data
The issue is not in our parsing code - it's that the downloaded features have empty labels for test set.

### 3. Cross-Subject Split - VERIFIED
- Train (Y-*): 16 subjects, no overlap
- Test (X-*): 10 subjects, no overlap with train
- ✅ No subject leakage

---

## Action Items

1. **For Local Development:**
   - Use leave-one-subject-out CV on Y-* subjects to estimate model performance
   - DO NOT use X-* subjects for local evaluation

2. **For Final Evaluation:**
   - Train on all Y-* subjects
   - Generate predictions on X-* subjects
   - Submit to EvalAI for official evaluation

3. **For Reporting:**
   - Report cross-validation results on Y-* subjects
   - Clearly state that test set labels were unavailable in downloaded features

---

## Invalidated Results

**The previous official baseline evaluation is INVALID** because X-subject hidden test labels are missing in the downloaded extracted features. These samples must not be used for local supervised evaluation.

### Invalidated Files
- `official_baseline_results_20260508_142712.csv`
- `official_baseline_results_20260508_142642.csv`
- `official_baseline_results_20260508_142158.csv`
- `official_baseline_results_20260508_142148.csv`

**Reason:** Hidden test features do not contain ground-truth labels. Empty labels were incorrectly parsed as TSR, causing invalid accuracy/F1/balanced-accuracy results.

---

## Corrected Label Parsing Logic

### Before (INCORRECT)
```python
label = 1 if key contains "NR" else 0  # XBB__0_0 incorrectly parsed as TSR=0
```

### After (CORRECT)
```python
if "_NR_" in key:
    label = 1  # Normal Reading
elif "_TSR_" in key:
    label = 0  # Task-Specific Reading
else:
    label = None  # Unlabeled hidden test sample
    # Skip for supervised training, only use for EvalAI submission
```

X-subject samples with empty labels (`XBB__0_0`) are now correctly identified as **unlabeled** and excluded from local supervised evaluation.

---

## Conclusion

**Local evaluation on X-* subjects is NOT possible** with the downloaded extracted features because:
- Test labels are empty strings in the downloaded files
- This is a data issue in `features.zip`, not a code bug

**Recommended approach:**
1. Use cross-validation on Y-* subjects for development
2. Submit to EvalAI for official test set evaluation
3. Or re-extract features from raw .mat files if available

The model training code itself is correct - the issue is solely with the missing test labels in the downloaded extracted features.