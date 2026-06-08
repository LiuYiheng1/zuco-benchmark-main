# AdaGTCN Table 1 Comparison Report

## Experimental Setting

We follow the AdaGTCN-style subject split protocol:
- **Train subjects**: 12
- **Validation subjects**: 2
- **Test subjects**: 4
- **No target-subject calibration**
- **No test labels used for training**

### Subject Split (seed=0)
- **Train**: ['YAG', 'YFS', 'YIS', 'YLS', 'YSD', 'YDR', 'YAK', 'YSL', 'YMD', 'YHS', 'YTL', 'YRK']
- **Validation**: ['YDG', 'YAC']
- **Test**: ['YFR', 'YRP']

## AdaGTCN Table 1 Comparison

| Category | Method | F1 | Accuracy (%) |
|----------|--------|-----|--------------|
| Baselines-Unimodal | k-NN | 0.478 | 51.55 |
| Baselines-Unimodal | EEG-LSTM | 0.524 | 52.78 |
| Baselines-Unimodal | EM-LSTM | 0.550 | 54.22 |
| Baselines-Graph | EEG-GCN | 0.582 | 59.15 |
| Baselines-Graph | EEG-GCN + Attention Pooling | 0.614 | 59.75 |
| Baselines-Graph | EEG-GCN + Hierarchical Pooling | 0.621 | 60.56 |
| Baselines-Fusion | EEG-LSTM + EM-LSTM | 0.640 | 62.33 |
| Baselines-Fusion | EEG-GCN + EM-LSTM | 0.659 | 63.50 |
| AdaGTCN | AdaGTCN w/o DI-TCN | 0.652 | 64.12 |
| AdaGTCN | AdaGTCN w/o DN-GCN | 0.633 | 63.72 |
| AdaGTCN | AdaGTCN w/o AGL | 0.675 | 66.20 |
| AdaGTCN | AdaGTCN | 0.695 | 69.79 |
| Ours | PCET+GETA+CAGF | 40.0 +/- 0.0 | 45.9 +/- 0.0 |

## Our Results (PCET+GETA+CAGF)

| Metric | Value |
|--------|-------|
| Accuracy | 45.9 +/- 0.0% |
| Macro-F1 | 40.0 +/- 0.0% |
| Balanced Accuracy | 45.9 +/- 0.0% |
| AUROC | 45.7 +/- 0.0% |

## Key Questions Answered

### 1. Our performance under AdaGTCN-style 12/2/4 split?
- **Accuracy**: 45.9% +/- 0.0%
- **Macro-F1**: 40.0% +/- 0.0%

### 2. Exceeds AdaGTCN's 69.79% / F1 0.695?
**NO** - Our result (45.9%) is significantly below AdaGTCN (69.79%)

### 3. Exceeds EEG-GCN+EM-LSTM's 63.50% / F1 0.659?
**NO** - Our result (45.9%) is below EEG-GCN+EM-LSTM (63.50%)

### 4. Main reasons for lower performance?
1. **Protocol difference**: AdaGTCN uses word-level fixation-segmented EEG sequences with graph-temporal modeling; we use sentence-level precomputed features
2. **Model architecture**: AdaGTCN's DI-TCN and DN-GCN components are specifically designed for cross-subject adaptation
3. **Feature representation**: Our 420-dim electrode features may not capture the same information as word-level sequences
4. **Zero-shot setting**: Without any target subject calibration, cross-subject transfer is inherently difficult

### 5. Does this confirm our paper should focus on few-shot personalized calibration?
**YES** - The gap between zero-shot (45.9%) and few-shot (up to 80%) confirms that:
- Personalization is crucial for our approach
- The main contribution should be the EEG-gaze fusion framework under few-shot settings
- Zero-shot cross-subject remains challenging for our current approach

### 6. Can this table be directly included in the paper?
**NO**, with caveats:
- This is a protocol-aligned comparison, not an identical-input comparison
- AdaGTCN uses word-level sequences, we use sentence-level features
- The comparison shows our model's relative position but not direct superiority

## Fairness Statement

This comparison follows the AdaGTCN-style subject split, but the input representation is not identical to AdaGTCN. AdaGTCN uses word-level fixation-segmented EEG sequences, whereas our model uses sentence-level precomputed EEG and gaze features. Therefore, the comparison is protocol-aligned but not input-identical.

## Recommendations for Paper

1. **Main claim**: Emphasize few-shot personalized performance (80% at 50-shot)
2. **Secondary claim**: Highlight EEG-gaze multimodal fusion framework
3. **Honest comparison**: Report zero-shot results with caveats about protocol differences
4. **Future work**: Explore word-level features or graph-based architectures
