# ZuCo Benchmark Protocol Audit Summary

## Audit Date
2026-05-19 01:28:29

---

## 1. Feature Label Leakage Audit

### Summary
| Feature Set | Samples | Input Dim | Last Column Unique | Last-Col Clf Acc | Shuffled Acc |
|-------------|---------|-----------|-------------------|------------------|--------------|
| electrode_features_all | 14073 | 420 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.713676731793961 |
| sent_gaze_sacc | 18454 | 9 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.708209157409916 |
| sent_gaze_sacc_eeg_means | 18444 | 13 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.6993765248034698 |
| eeg_means | 18444 | 4 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.6993765248034698 |
| sent_gaze | 18454 | 4 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.708209157409916 |
| theta_mean | 18444 | 1 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.6993765248034698 |
| alpha_mean | 18444 | 1 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.6993765248034698 |
| beta_mean | 18444 | 1 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.6993765248034698 |
| gamma_mean | 18444 | 1 | ['', 'NR', 'TSR'] | N/A (not binary) | 0.6993765248034698 |

### Key Findings
- Last column contains labels (NR/TSR) for all feature sets
- Model inputs correctly use `value[:-1]` for features
- Shuffled label sanity check confirms no data leakage when labels are randomized

---

## 2. Split Audit

### Subject Groups
- **Y Subjects (Training/Development):** 16 subjects
  ```
  YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL
  ```
- **X Subjects (Heldout/Test):** 10 subjects
  ```
  XBB, XDT, XLS, XPB, XSE, XTR, XWS, XAH, XBD, XSS
  ```
- **Overlap between Y and X:** 0 subjects

### X Subjects Label Check
| Subject | Has Features | Last Column | Appears to be Label |
|---------|--------------|-------------|---------------------|
| XBB | True |  | True |
| XDT | True |  | True |
| XLS | True |  | True |
| XPB | True |  | True |
| XSE | True |  | True |
| XTR | True |  | True |
| XWS | True |  | True |
| XAH | True |  | True |
| XBD | True |  | True |
| XSS | True |  | True |

---

## 3. EEG-Gaze Alignment Audit

| Metric | Count |
|--------|-------|
| EEG keys (electrode_features_all) | 14073 |
| Gaze keys (sent_gaze_sacc) | 18454 |
| Matching keys | 7 |
| EEG-only keys | 14066 |
| Gaze-only keys | 18447 |

### Unmatched Keys
- EEG-only sample: YIS_TSR_57_3672
- Gaze-only sample: YRK_NR_85_6454

---

## 4. Duplicate Recovery Audit

| Metric | Count |
|--------|-------|
| NR sentences loaded | 318 |
| TSR sentences loaded | 324 |
| **NR-TSR duplicate pairs found** | **43** |
| Expected (from paper) | 63 |

### Status
❌ FAIL: Duplicate count significantly different from expected

---

## 5. Feature Overlap Audit

### Feature Dimensions
| Feature Set | Dimension |
|-------------|-----------|
| theta_mean | 1-D |
| alpha_mean | 1-D |
| beta_mean | 1-D |
| gamma_mean | 1-D |
| eeg_means | 4-D |
| sent_gaze | 4-D |
| sent_saccade | 6-D |
| sent_gaze_sacc | 9-D |
| sent_gaze_sacc_eeg_means | 13-D |
| electrode_features_all | 420-D |

### sent_gaze_sacc_eeg_means Composition
- sent_gaze_sacc (9-D) + eeg_means (4-D) = **13-D** ✓

### Problematic Combinations (Avoid)
- `sent_gaze_sacc` + `sent_gaze_sacc_eeg_means` → 9 features duplicated
- `sent_gaze` + `sent_gaze_sacc` → sent_gaze is subset
- `eeg_means` + `sent_gaze_sacc_eeg_means` → 4 features duplicated

### Recommended Clean Combinations
- **EEG-only:** electrode_features_all (420-D)
- **Gaze-only:** sent_gaze_sacc (9-D)
- **EEG+Gaze:** electrode_features_all + sent_gaze_sacc (429-D)

---

## 6. LOSO Baseline Results

### Per-Subject Results
| Subject | Model | Accuracy | Balanced Acc | Macro-F1 | AUROC |
|---------|-------|----------|--------------|----------|-------|
| YAC | EEG-only | 0.4389 | 0.5000 | 0.3050 | 0.7345 |
| YAG | EEG-only | 0.5502 | 0.5148 | 0.3810 | 0.5568 |
| YAK | EEG-only | 0.4714 | 0.4957 | 0.4674 | 0.5494 |
| YDG | EEG-only | 0.6540 | 0.6560 | 0.6536 | 0.7254 |
| YDR | EEG-only | 0.5000 | 0.5516 | 0.4447 | 0.6399 |
| YFR | EEG-only | 0.4971 | 0.5003 | 0.4961 | 0.5301 |
| YFS | EEG-only | 0.5594 | 0.5113 | 0.5010 | 0.5556 |
| YHS | EEG-only | 0.5077 | 0.4910 | 0.3466 | 0.5496 |
| YIS | EEG-only | 0.5652 | 0.5566 | 0.5530 | 0.6068 |
| YLS | EEG-only | 0.4085 | 0.5018 | 0.2930 | 0.5606 |
| YMD | EEG-only | 0.6852 | 0.6848 | 0.6820 | 0.7428 |
| YRK | EEG-only | 0.5128 | 0.4959 | 0.3390 | 0.4840 |
| YRP | EEG-only | 0.5401 | 0.5474 | 0.5306 | 0.5979 |
| YSD | EEG-only | 0.5652 | 0.5866 | 0.5346 | 0.7002 |
| YSL | EEG-only | 0.5398 | 0.5106 | 0.4132 | 0.3912 |
| YTL | EEG-only | 0.4605 | 0.4575 | 0.4569 | 0.4982 |
| YAC | Gaze-only | 0.5509 | 0.5472 | 0.5450 | 0.5694 |
| YAG | Gaze-only | 0.5169 | 0.5413 | 0.4201 | 0.5616 |
| YAK | Gaze-only | 0.4181 | 0.4180 | 0.4177 | 0.3711 |
| YDG | Gaze-only | 0.6888 | 0.6803 | 0.6770 | 0.7968 |
| YDR | Gaze-only | 0.5968 | 0.5747 | 0.5058 | 0.7608 |
| YFR | Gaze-only | 0.6312 | 0.6319 | 0.6278 | 0.6856 |
| YFS | Gaze-only | 0.6103 | 0.5937 | 0.5622 | 0.7669 |
| YHS | Gaze-only | 0.5453 | 0.5189 | 0.3899 | 0.7938 |
| YIS | Gaze-only | 0.8796 | 0.8821 | 0.8796 | 0.9327 |
| YLS | Gaze-only | 0.5993 | 0.5953 | 0.5934 | 0.6542 |
| YMD | Gaze-only | 0.6076 | 0.6259 | 0.5698 | 0.7413 |
| YRK | Gaze-only | 0.4953 | 0.5159 | 0.4276 | 0.5440 |
| YRP | Gaze-only | 0.4574 | 0.4741 | 0.4126 | 0.4575 |
| YSD | Gaze-only | 0.8241 | 0.8284 | 0.8239 | 0.9266 |
| YSL | Gaze-only | 0.5115 | 0.5364 | 0.4072 | 0.6626 |
| YTL | Gaze-only | 0.9026 | 0.9066 | 0.9025 | 0.9698 |
| YAC | Gaze+EEG-mean | 0.5489 | 0.5379 | 0.5058 | 0.5928 |
| YAG | Gaze+EEG-mean | 0.5453 | 0.5671 | 0.4771 | 0.5711 |
| YAK | Gaze+EEG-mean | 0.4499 | 0.4435 | 0.4387 | 0.3953 |
| YDG | Gaze+EEG-mean | 0.6319 | 0.6507 | 0.5951 | 0.7997 |
| YDR | Gaze+EEG-mean | 0.6103 | 0.5890 | 0.5290 | 0.7741 |
| YFR | Gaze+EEG-mean | 0.6246 | 0.6327 | 0.6237 | 0.7021 |
| YFS | Gaze+EEG-mean | 0.5995 | 0.5803 | 0.5329 | 0.7596 |
| YHS | Gaze+EEG-mean | 0.5304 | 0.5029 | 0.3517 | 0.8033 |
| YIS | Gaze+EEG-mean | 0.8417 | 0.8387 | 0.8401 | 0.9276 |
| YLS | Gaze+EEG-mean | 0.5943 | 0.6138 | 0.5941 | 0.6733 |
| YMD | Gaze+EEG-mean | 0.6441 | 0.6595 | 0.6222 | 0.7690 |
| YRK | Gaze+EEG-mean | 0.4926 | 0.5106 | 0.4432 | 0.5557 |
| YRP | Gaze+EEG-mean | 0.4641 | 0.4881 | 0.3592 | 0.5018 |
| YSD | Gaze+EEG-mean | 0.5629 | 0.5859 | 0.4884 | 0.9019 |
| YSL | Gaze+EEG-mean | 0.5034 | 0.5292 | 0.3870 | 0.6906 |
| YTL | Gaze+EEG-mean | 0.9188 | 0.9216 | 0.9188 | 0.9689 |

### Aggregated Results
| Model | Accuracy (mean±std) | Balanced Acc (mean±std) | Macro-F1 (mean±std) | AUROC (mean±std) |
|-------|---------------------|-------------------------|---------------------|------------------|
| EEG-only | 0.5285 ± 0.0719 | 0.5351 ± 0.0615 | 0.4623 ± 0.1146 | 0.5889 ± 0.0989 |
| Gaze-only | 0.6147 ± 0.1438 | 0.6169 ± 0.1421 | 0.5726 ± 0.1714 | 0.6997 ± 0.1708 |
| Gaze+EEG-mean | 0.5977 ± 0.1258 | 0.6032 ± 0.1245 | 0.5442 ± 0.1575 | 0.7117 ± 0.1589 |

---

## Audit Conclusion

### Overall Status
❌ **ISSUES FOUND**

### Issues to Address:
- Duplicate recovery: Found 43 but expected ~63
- Alignment: Significant mismatch (14066 EEG-only, 18447 gaze-only)

---

*Generated by ZuCo Benchmark Protocol Audit Script*