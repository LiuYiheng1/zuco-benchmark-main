# Literature Comparison Report

## Reported vs Our Results

| Method | Reported Acc | Reported F1 | Our Acc | Our F1 | Protocol |
|--------|--------------|-------------|---------|--------|----------|
| Random | 50.00 | 50.00 | - | - | ZuCo Benchmark |
| BERT baseline | 65.00 | 64.00 | - | - | ZuCo Benchmark |
| Eye-tracking baseline | 69.00 | 67.00 | - | - | ZuCo Benchmark |
| Eye-tracking + EEG mean | 68.00 | 66.00 | - | - | ZuCo Benchmark |
| EEG electrode + PCA | 58.00 | 56.00 | - | - | ZuCo Benchmark |
| k-NN | 51.55 | 47.80 | - | - | AdaGTCN |
| EEG-LSTM | 52.78 | 52.40 | - | - | AdaGTCN |
| EM-LSTM | 54.22 | 55.00 | - | - | AdaGTCN |
| EEG-GCN | 59.15 | 58.20 | - | - | AdaGTCN |
| EEG-GCN+EM-LSTM | 63.50 | 65.90 | - | - | AdaGTCN |
| AdaGTCN | 69.79 | 69.50 | - | - | AdaGTCN |
| **Ours (PCET+GETA+CAGF)** | - | - | **51.5 +/- 8.6** | **43.2** | Benchmark-style LOSO |
| **Ours (PCET+GETA+CAGF)** | - | - | **45.9** | **40.0** | AdaGTCN-style 12/2/4 |

## Key Findings

### 1. Benchmark-style LOSO (Zero-shot Cross-subject)
- Our model achieves **51.5 +/- 8.6%** accuracy

### 2. AdaGTCN-style 12/2/4 Split
- Our model achieves **45.9%** accuracy

### 3. Comparison with Literature
- vs ZuCo Benchmark eye-tracking baseline (69.0%): BELOW (51.5% vs 69.0%)
- vs AdaGTCN reported (69.79%): BELOW (51.5% vs 69.79%)

### 4. Few-shot vs Zero-shot Gap
- Few-shot personalized results (3/5/10/20/50-shot) range from 62-80%
- Zero-shot cross-subject LOSO achieves ~51.5%
- The gap shows significant benefit from subject-specific calibration

## Notes
- Reported results from different original protocols are NOT directly comparable.
- Our zero-shot LOSO is the most stringent evaluation protocol.
- Few-shot personalized results demonstrate practical utility.
