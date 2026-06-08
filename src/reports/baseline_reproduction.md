# ZuCo 2.0 Baseline Reproduction Report

## Experiment Setup
- **Date**: 2026-06-03 13:08:46
- **Split Protocol**: Cross-subject (train on Y-subjects, test on X-subjects)
- **Seeds**: [0, 1, 2, 3, 4]

## Feature Sets
1. **EEG-only**: electrode_features_all (420 features)
2. **Gaze-only**: sent_gaze_sacc (10 features)
3. **Combined**: sent_gaze_sacc_eeg_means (14 features)

## Train Subjects (16)
YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL

## Test Subjects (10)
XBB, XDT, XLS, XPB, XSE, XTR, XWS, XAH, XBD, XSS

## Results

| Feature Set | Accuracy | Macro-F1 | Balanced Accuracy |
|-------------|----------|----------|------------------|
| electrode_features_all | 0.6602 ¡À 0.0000 | 0.3977 ¡À 0.0000 | 0.6602 ¡À 0.0000 |
| sent_gaze_sacc | 0.3914 ¡À 0.0000 | 0.2813 ¡À 0.0000 | 0.3914 ¡À 0.0000 |
| sent_gaze_sacc_eeg_means | 0.3950 ¡À 0.0000 | 0.2832 ¡À 0.0000 | 0.3950 ¡À 0.0000 |

## Notes
- Labels: NR (Normal Reading) = 1, TSR (Task-Specific Reading) = 0
- Cross-subject protocol: No subject overlap between train and test
- Evaluation: Per-sample prediction aggregated across all test subjects

## Command
```bash
cd src
python run_official_baseline.py
```
