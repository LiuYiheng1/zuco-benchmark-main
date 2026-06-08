# AdaGTCN-inspired 10/2/4 Split Debug Report

## Important Note
**This is a diagnostic experiment, NOT a main paper result.**

## 1. Available Subjects
- **Total Y-subjects**: 16
- **Cannot run strict 12/2/4** (only 16 available)
- **Using**: AdaGTCN-inspired **10/2/4** split

## 2. Subject Split (seed=0)
- **Train**: ['YAG', 'YFS', 'YIS', 'YLS', 'YSD', 'YDR', 'YAK', 'YSL', 'YMD', 'YHS'] (10 subjects)
- **Val**: ['YTL', 'YRK'] (2 subjects)
- **Test**: ['YDG', 'YAC', 'YFR', 'YRP'] (4 subjects)

## 3. Class Distribution
| Set | NR (label=1) | TSR (label=0) | NR Ratio |
|-----|--------------|----------------|----------|
| Train | 2814 | 3387 | 45.4% |
| Test | 766 | 857 | 47.2% |

## 4. Class Order Analysis

All models have **classes_ = [0, 1]** where:
- class 0 = TSR
- class 1 = NR

**No class order issue found.**

## 5. Results Summary (Single Split, seed=0)

| Method | Accuracy | Macro-F1 | Balanced Acc | AUROC | Inverted Acc |
|--------|----------|----------|--------------|-------|-------------|
| Majority | 52.8% | - | - | - | - |
| Random | 50.3% | 50.2% | ~50% | ~0.50 | - |
| EEG_SVM | 54.2% | 53.4% | 55.1% | 57.3% | 45.8% |
| Gaze_SVM | 58.8% | 58.7% | 58.7% | 64.2% | 41.2% |
| PCET_source | 53.0% | 52.2% | 53.9% | 57.5% | 47.0% |
| GETA_source | 54.7% | 53.9% | 55.5% | 57.5% | 45.3% |
| CAGF | 53.0% | 52.1% | 53.9% | 57.5% | 47.0% |

## 6. Confusion Matrices

### Random
```
[[432 425]
 [382 384]]
```

### EEG_SVM
```
[[334 523]
 [220 546]]
```

### Gaze_SVM
```
[[526 331]
 [337 429]]
```

### PCET_source
```
[[329 528]
 [235 531]]
```

### GETA_source
```
[[340 517]
 [219 547]]
```

### CAGF
```
[[321 536]
 [227 539]]
```

## 7. CAGF Alpha Analysis
- **Mean**: 0.4969
- **Std**: 0.0167
- **Min**: 0.4198
- **Max**: 0.5501

**Alpha is NOT collapsed** - showing reasonable variation around 0.5.

## 8. Key Questions Answered

1. **Available subjects**: 16 Y-subjects
2. **Can run 12/2/4**: NO - using 10/2/4 instead
3. **Split used**: AdaGTCN-inspired **10/2/4**
4. **Class order issue**: NO - classes_ = [0, 1] consistently (0=TSR, 1=NR)
5. **CAGF label issue**: NO - both interpretations give similar results
6. **Random Macro-F1**: 50.2% (near-balanced data)
7. **Cross-subject vs few-shot gap**: ~59% vs ~80% (gap of ~21%)

## 9. Conclusions

- **No class order bug** - label mapping is consistent
- **All models perform close to majority/random** (~50%) in zero-shot setting
- **CAGF fusion does not significantly improve** over individual models
- **Cross-subject transfer is inherently difficult** for this task
- **Few-shot personalized calibration is essential** for good performance
- **Current results are diagnostic only**, not suitable for main paper

## 10. Recommendations

1. **Main paper claim**: Use few-shot personalized results (up to 80%)
2. **Cross-subject results**: Report honestly as baseline, with caveats
3. **Future work**: Explore domain adaptation or subject-specific calibration
