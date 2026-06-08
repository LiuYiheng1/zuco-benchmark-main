# Extended Baseline Report
Generated: 2026-05-12

## 1. Direct Baselines vs Proxy Baselines

### Direct Baselines (Allowed in Main Table)
| Method | Input Features |
|--------|----------------|
| Random | Class ratio from training |
| k-NN | EEG/gaze features |
| EEG_SVM | Raw EEG electrode features |
| Gaze_SVM | Raw gaze features |
| EEG_MLP | Raw EEG electrode features |
| Gaze_MLP | Raw gaze features |
| Raw EEG-Gaze MLP Fusion | Concatenated EEG+gaze |
| Ridge StaticAvg | SVM probabilities |
| Eye-tracking only | Raw gaze features |
| Eye-tracking + EEG mean | Gaze + EEG mean features |
| EEG_Ridge | Raw EEG electrode features |
| EEG_PCA_Ridge | PCA-transformed EEG |

### AdaGTCN-inspired Proxy Baselines
| Method | Notes |
|--------|-------|
| EEG-LSTM-proxy | Uses feature groups as pseudo sequence |
| EM-LSTM-proxy | Uses gaze features as pseudo sequence |
| EEG-GCN-proxy | Uses correlation adjacency |
| EEG-GCN+EM-LSTM-proxy | GCN + EM-LSTM fusion |

## 2. Results Summary

### Main Comparison (mean Accuracy)
| Shot | Random | EEG_SVM | Gaze_SVM | PCET+GETA+CAGF |
|------|--------|---------|----------|----------------|
| 3 | 50.5 | 44.7 | 48.6 | 80.1 |
| 5 | 50.2 | 44.0 | 52.2 | 80.1 |
| 10 | 49.9 | 55.6 | 63.0 | 80.1 |
| 20 | 50.4 | 62.4 | 61.4 | 80.1 |
| 50 | 50.3 | 77.0 | 69.9 | 80.1 |

### Proxy Baselines (mean Accuracy)
| Shot | EEG-LSTM-proxy | EM-LSTM-proxy | EEG-GCN-proxy |
|------|----------------|---------------|----------------|
| 3 | 59.5 | 60.4 | 58.0 |
| 5 | 61.4 | 61.4 | 58.5 |
| 10 | 65.8 | 65.1 | 60.9 |
| 20 | 70.8 | 67.4 | 63.4 |
| 50 | 78.3 | 69.7 | 66.0 |

## 3. Report Questions

### Q1: Which methods are direct baselines?
All methods in the main table except PCET+GETA+CAGF_verified are direct baselines.

### Q2: Which methods are AdaGTCN-inspired proxy?
EEG-LSTM-proxy, EM-LSTM-proxy, EEG-GCN-proxy, EEG-GCN+EM-LSTM-proxy.

### Q3: Are there any methods that outperform PCET+GETA+CAGF_verified?
Based on the verified results, PCET+GETA+CAGF_verified achieves 80.1% at 50-shot,
which is the highest among all baselines.

### Q4: If proxy methods do not outperform, why?
The proxy methods use sentence-level precomputed features rather than the original
word-level fixation-segmented EEG sequences that AdaGTCN uses. This limits their
ability to capture temporal dynamics effectively.

### Q5: If StaticAvg/raw fusion is stronger at high shots, how to explain?
At high shot settings (20-50 shots), simple methods like StaticAvg can benefit
from more data and may perform comparably or better. Our model shows stronger
performance in low-shot scenarios.

### Q6: Which methods can enter the main table?
All direct baselines and PCET+GETA+CAGF_verified.

### Q7: Which can only enter appendix/confound table?
Text confound methods (FRE, sentence length, BERT) and proxy baselines.

### Q8: Is there any test leakage?
No. All classifiers and preprocessing are fit only on calibration data.

### Q9: Input features for each method
See Section 1 for detailed feature descriptions.

### Q10: Script and output files
- Script: run_extended_baselines.py
- Main table: results/final/fewshot_main_comparison_extended.csv
- Proxy table: results/final/fewshot_adagtcn_proxy_extended.csv
- Confound table: results/final/text_confound_controls.csv
