# ZuCo 2.0 LOSO-Y Cross-Validation Report

## Date: 2026-05-08

## Protocol
- **Method**: Leave-One-Subject-Out on Y-subjects only (LOSO-Y)
- **Folds**: 16 (one hold-out per Y-subject)
- **Seeds**: [0]
- **Model**: SVM (linear kernel, gamma='scale')
- **Features**: EEG electrode_features_all (420 features)
- **X-subjects**: Excluded from local evaluation (hidden test, no ground-truth labels)

## LOSO-Y SVM EEG Full Results

| Subject | Accuracy | Macro-F1 | Balanced Accuracy | Test Samples |
|---------|----------|----------|-------------------|--------------|
| YAC | 0.4389 | 0.3050 | 0.5000 | 360 |
| YAG | 0.5699 | 0.4242 | 0.5361 | 658 |
| YAK | 0.4246 | 0.2981 | 0.5000 | 577 |
| YDG | **0.6369** | **0.6347** | **0.6349** | 526 |
| YDR | 0.4757 | 0.3996 | 0.5323 | 618 |
| YFR | 0.4971 | 0.4786 | 0.4895 | 350 |
| YFS | 0.5840 | 0.5528 | 0.5532 | 488 |
| YHS | 0.5397 | 0.4933 | 0.5298 | 717 |
| YIS | 0.6104 | 0.5925 | 0.5990 | 729 |
| YLS | 0.4128 | 0.3009 | 0.5054 | 470 |
| YMD | 0.5833 | 0.5132 | 0.5819 | 540 |
| YRK | 0.5171 | 0.3408 | 0.5000 | 234 |
| YRP | 0.4315 | 0.4117 | 0.4404 | 387 |
| YSD | 0.5568 | 0.5169 | 0.5803 | 713 |
| YSL | 0.5326 | 0.3475 | 0.4986 | 691 |
| YTL | 0.4261 | 0.4217 | 0.4227 | 697 |

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | 0.5148 | 0.0731 | 0.4128 | 0.6369 |
| Macro-F1 | 0.4391 | 0.0992 | 0.2981 | 0.6347 |
| Balanced Accuracy | 0.5148 | 0.0577 | 0.4227 | 0.6349 |

## Key Observations

### 1. High Subject Variability
- Best performer: YDG (Acc=0.64, F1=0.63)
- Worst performer: YLS (Acc=0.41, F1=0.30)
- Significant std across subjects (0.07 for accuracy)

### 2. Some Subjects at Chance Level
- YAC, YAK, YRK: Balanced Accuracy ≈ 0.50 (chance level)
- This suggests EEG patterns for these subjects don't generalize well

### 3. Majority Class Baseline
- Train data is balanced (~50% NR, ~50% TSR)
- Majority baseline would predict ~50% accuracy
- **Our SVM achieves 51.5% mean accuracy** - barely above chance

### 4. Label Distribution Check
- Test NR ratio: 0.40-0.50 (similar to train)
- Balanced dataset confirms no label imbalance issue

## Conclusions

1. ✅ **LOSO-Y Protocol Correctly Implemented**
   - Only labeled Y-subjects used
   - No X-subject leakage
   - Cross-subject split verified

2. ⚠️ **EEG-only SVM Baseline Performance**
   - Mean accuracy: 51.5% (barely above chance)
   - High variance across subjects
   - Suggests EEG patterns are subject-specific

3. 📊 **Ready for TGCR Comparison**
   - Now can compare TGCR against this baseline
   - If TGCR significantly outperforms 51.5%, it shows value

## Files Generated

- `results/loso/svm_eeg_loso_all_20260508_164746.csv` - Full results
- `results/loso/loso_log.txt` - Execution log

## Next Steps

1. Run LOSO-Y with multiple seeds [0, 1, 2, 3, 4]
2. Implement and run PyTorch baselines (MLP, Late Fusion, Attention)
3. Implement and run TGCR v1
4. Compare all models against SVM baseline
5. Generate EvalAI submissions for X-subject predictions