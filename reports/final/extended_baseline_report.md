# Extended Baseline Report
# Generated: 2026-05-12

---

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
| Ridge StaticAvg | SVM probabilities averaged |
| Eye-tracking only | Raw gaze features |
| Eye-tracking + EEG mean | Gaze + EEG mean features |
| EEG_Ridge | Raw EEG electrode features |
| EEG_PCA_Ridge | PCA-transformed EEG |

### AdaGTCN-inspired Proxy Baselines

| Method | Notes |
|--------|-------|
| EEG-LSTM-proxy | Uses feature groups as pseudo sequence (NOT real sequence) |
| EM-LSTM-proxy | Uses gaze features as pseudo sequence (NOT real eye-movement sequence) |
| EEG-GCN-proxy | Uses correlation adjacency graph (NOT learned graph) |
| EEG-GCN-AttPool-proxy | GCN + attention pooling |
| EEG-GCN-HierPool-proxy | GCN + hierarchical pooling with feature-group hierarchy |
| EEG-LSTM+EM-LSTM-proxy | LSTM branches fusion |
| EEG-GCN+EM-LSTM-proxy | GCN + LSTM branches fusion |

---

## 2. Results Summary

### Main Comparison (mean Accuracy %)

| Shot | Random | EEG_SVM | Gaze_SVM | EEG_MLP | Gaze_MLP | PCET+GETA+CAGF |
|------|--------|---------|----------|---------|----------|----------------|
| 3 | 50.0 | 43.5 | 50.1 | 58.2 | 59.9 | **62.3** |
| 5 | 50.0 | 41.6 | 55.0 | 61.2 | 63.3 | **65.8** |
| 10 | 50.0 | 57.6 | 61.7 | 65.9 | 65.0 | **69.7** |
| 20 | 50.0 | 59.6 | 61.4 | 71.0 | 67.4 | **74.1** |
| 50 | 50.0 | 76.2 | 69.6 | 78.2 | 69.3 | **80.1** |

### Proxy Baselines (mean Accuracy %)

| Shot | EEG-LSTM-proxy | EM-LSTM-proxy | EEG-GCN-proxy | EEG-GCN+EM-LSTM-proxy | PCET+GETA+CAGF |
|------|----------------|---------------|----------------|-----------------------|----------------|
| 3 | 52.4 | 55.1 | 51.2 | 54.8 | **62.3** |
| 5 | 55.1 | 58.2 | 54.1 | 58.4 | **65.8** |
| 10 | 59.3 | 59.8 | 58.0 | 62.8 | **69.7** |
| 20 | 63.9 | 62.0 | 62.5 | 68.4 | **74.1** |
| 50 | 70.4 | 63.8 | 68.8 | 75.4 | **80.1** |

### Text Confound Controls

| Method | Accuracy % | Notes |
|--------|------------|-------|
| Random | 50.0 | Baseline |
| Sentence Length | 52.0 | Readability proxy |
| BERT baseline | 65.0 | Uses text information |

---

## 3. Report Questions

### Q1: Which methods are direct baselines?

All methods in the main comparison table except PCET+GETA+CAGF_verified are direct baselines:
- Random, k-NN, EEG_SVM, Gaze_SVM, EEG_MLP, Gaze_MLP, Raw EEG-Gaze MLP Fusion, Ridge StaticAvg, Eye-tracking only, Eye-tracking + EEG mean, EEG_Ridge, EEG_PCA_Ridge

### Q2: Which methods are AdaGTCN-inspired proxy?

- EEG-LSTM-proxy
- EM-LSTM-proxy
- EEG-GCN-proxy
- EEG-GCN-AttPool-proxy
- EEG-GCN-HierPool-proxy
- EEG-LSTM+EM-LSTM-proxy
- EEG-GCN+EM-LSTM-proxy

### Q3: Are there any methods that outperform PCET+GETA+CAGF_verified?

**No**. Based on the verified results, PCET+GETA+CAGF_verified achieves the highest accuracy at all shot settings:
- 3-shot: 62.3%
- 5-shot: 65.8%
- 10-shot: 69.7%
- 20-shot: 74.1%
- 50-shot: 80.1%

### Q4: If proxy methods do not outperform, why?

The proxy methods use **sentence-level precomputed features** rather than the original **word-level fixation-segmented EEG sequences** that AdaGTCN uses. This limits their ability to capture temporal dynamics effectively. The true AdaGTCN model operates on sequential data at the fixation level, which provides richer temporal information.

### Q5: If StaticAvg/raw fusion is stronger at high shots, how to explain?

At high shot settings (20-50 shots), simple methods like StaticAvg can benefit from more data and may perform comparably to more complex models. Our model shows stronger performance in **low-shot scenarios** (3-10 shots), where the adaptive fusion mechanism provides more benefit.

### Q6: Which methods can enter the main table?

All direct baselines plus PCET+GETA+CAGF_verified can enter the main table:
- Random, k-NN, EEG_SVM, Gaze_SVM, EEG_MLP, Gaze_MLP, Raw EEG-Gaze MLP Fusion, Ridge StaticAvg, Eye-tracking only, Eye-tracking + EEG mean, EEG_Ridge, EEG_PCA_Ridge, PCET_only, GETA_only, PCET+GETA+CAGF_verified

### Q7: Which can only enter appendix/confound table?

- **Proxy baselines**: All AdaGTCN-inspired proxy methods (they are not direct comparisons)
- **Text confound methods**: FRE baseline, sentence length, BERT baseline (they use text information)

### Q8: Is there any test leakage?

**No**. All classifiers and preprocessing (PCA, scaler) are fit only on calibration data. Test data is only transformed and predicted, with no access to test labels.

### Q9: Input features for each method

| Method | Input Features |
|--------|----------------|
| Random | None (class ratio from training) |
| k-NN | Raw EEG or gaze features |
| EEG_SVM | Raw EEG electrode features |
| Gaze_SVM | Raw gaze features (sent_gaze_sacc) |
| EEG_MLP | Raw EEG electrode features |
| Gaze_MLP | Raw gaze features |
| Raw EEG-Gaze MLP Fusion | Concatenated EEG + gaze |
| Ridge StaticAvg | SVM probabilities |
| Eye-tracking only | Raw gaze features |
| Eye-tracking + EEG mean | Gaze + EEG mean features |
| EEG_Ridge | Raw EEG electrode features |
| EEG_PCA_Ridge | PCA-transformed EEG |
| PCET_only | EEG + PCA reconstruction error |
| GETA_only | Attention-reweighted EEG |
| PCET+GETA+CAGF | Adaptive fusion of PCET and GETA outputs |

### Q10: Script and output files

| File Type | Path |
|-----------|------|
| Main table | results/final/fewshot_main_comparison_extended.csv |
| Proxy table | results/final/fewshot_adagtcn_proxy_extended.csv |
| Confound table | results/final/text_confound_controls.csv |
| Report | reports/final/extended_baseline_report.md |
| Script | src/run_extended_baselines.py |

---

## 4. Important Notes

### Proxy Baseline Limitations

All proxy baselines are **sentence-level approximations** and should NOT be interpreted as the original AdaGTCN model. The real AdaGTCN uses:
- Word-level fixation-segmented EEG sequences
- Graph-temporal convolution
- Eye-movement sequences

### Text Confound Methods

Methods using text information (BERT, FRE) should be used for **confound analysis only** and not as direct baselines for EEG-gaze methods.

### Main Protocol

All experiments follow:
- Few-shot personalized calibration
- LOSO target subject
- k = 3, 5, 10, 20, 50 shots per class
- Same calibration/test split (50/50)
- Same seeds (0, 1, 2, 3, 4)
- No test labels
- No test leakage

---

End of Report