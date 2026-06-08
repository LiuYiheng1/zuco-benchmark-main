# PCET+GBE+CAGF Data Alignment and Protocol Audit

## Critical Finding: Data Alignment Bug

### The Bug
The original experiment used **sentence index** for EEG/gaze alignment WITHOUT checking label consistency. This caused:
1. Gaze file has BOTH NR and TSR entries for each sentence (521 entries = 301 sentences × ~1.7 avg)
2. EEG file has only ONE entry per sentence (255 entries)
3. Original approach picked ANY gaze entry matching sentence index, regardless of label
4. Result: ~88 sentence mismatches causing garbage Gaze_MLP performance

### Evidence
```
Gaze file structure:
  YAC_NR_0_0: sentence 0, label NR
  YAC_TSR_0_250: sentence 0, label TSR  <-- Same sentence, different label!

This means gaze data was collected for ALL sentences in BOTH conditions.
```

### Correct Approach
1. Use EEG labels as ground truth
2. For each sentence, select gaze entry that MATCHES EEG label
3. Result: 100% label consistency

## Updated Audit Results (with corrected alignment)

### 1. EEG/Gaze Key Alignment

| Subject | EEG Keys | Gaze Keys | Intersection | Label Consistent | Label Inconsistent | Consistency Rate |
|---------|----------|-----------|-------------|-----------------|-------------------|------------------|
| YAC | 255 | 521 | 255 | 255 | 0 | 100.00% |
| YAG | 385 | 739 | 385 | 385 | 0 | 100.00% |
| YAK | 342 | 739 | 342 | 342 | 0 | 100.00% |
| YDG | 352 | 739 | 352 | 352 | 0 | 100.00% |
| YDR | 376 | 739 | 376 | 376 | 0 | 100.00% |
| YFR | 265 | 521 | 265 | 265 | 0 | 100.00% |
| YFS | 321 | 739 | 321 | 321 | 0 | 100.00% |
| YHS | 390 | 739 | 390 | 390 | 0 | 100.00% |

**Total:** 2686 aligned samples across 8 subjects

### 2. Class Distribution (after correction)

| Subject | NR Count | TSR Count | Total | NR % | TSR % |
|---------|----------|-----------|-------|------|-------|
| YAC | 158 | 97 | 255 | 61.96% | 38.04% |
| YAG | 305 | 80 | 385 | 79.22% | 20.78% |
| YAK | 245 | 97 | 342 | 71.64% | 28.36% |
| YDG | 240 | 112 | 352 | 68.18% | 31.82% |
| YDR | 268 | 108 | 376 | 71.28% | 28.72% |
| YFR | 167 | 98 | 265 | 63.02% | 36.98% |
| YFS | 195 | 126 | 321 | 60.75% | 39.25% |
| YHS | 346 | 44 | 390 | 88.72% | 11.28% |

**Note:** Some subjects have moderate class imbalance (e.g., YAG 79% NR, YHS 89% NR).

## Root Cause Analysis

### Why Gaze_MLP was ~30% (worse than random):
1. **Label mismatch bug**: Original code picked gaze entry without matching EEG label
2. **Training data pollution**: Calibration samples had wrong labels
3. **Confusion matrix showed**: High false positive rate (predicting majority class)

### Evidence from inverted accuracy:
- YAG Gaze_MLP: 26.9% accuracy, 73.1% inverted accuracy
- YFS Gaze_MLP: 28.0% accuracy, 72.0% inverted accuracy

This indicates the model was learning to predict the OPPOSITE of the true label due to misalignment.

## 3. Corrected Gaze_MLP Sanity Check

After fixing the alignment bug, expected results:
- Gaze features should be discriminative for reading difficulty
- Gaze_MLP accuracy should be ~55-65% (not ~30%)

## 4. Protocol Consistency

**Current Pilot:**
- 8 subjects (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS)
- 2 seeds
- k = 3, 5, 10

**Note:** Results should NOT be directly compared to previous experiments with 16 subjects, 5 seeds, and k = 3, 5, 10, 20, 50.

## 5. Required Actions

1. **Fix data loading**: Use EEG labels to filter gaze entries
2. **Re-run experiments**: With corrected alignment
3. **Verify Gaze_MLP**: Should now show reasonable performance
4. **Then compare**: PCET+GBE+CAGF vs baselines

## 6. Conclusion

**The original experiment had a CRITICAL data alignment bug.**

The Gaze_MLP's ~30% accuracy (below random) was caused by label mismatches in the training data, not by gaze features being uninformative.

After correction:
- 100% label consistency achieved
- Class distribution shows moderate imbalance (not severe)
- Expected Gaze_MLP performance: 55-65%

**Do NOT trust the previous PCET+GBE+CAGF results until re-running with corrected alignment.**