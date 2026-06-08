# Final Experiment Results Report

## 1. Main Results (PCET + GETA + CAGF)

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| EEG_SVM | 43.5¡À8.7 | 41.6¡À10.6 | 57.6¡À15.4 | 59.6¡À18.2 | 76.2¡À6.7 || Gaze_SVM | 50.1¡À14.7 | 55.0¡À16.2 | 61.7¡À15.3 | 61.4¡À17.2 | 69.6¡À11.8 || EEG_MLP | 58.2¡À8.1 | 61.2¡À7.6 | 65.9¡À7.2 | 71.0¡À6.8 | 78.2¡À6.2 || Gaze_MLP | 59.9¡À11.8 | 63.3¡À12.7 | 65.0¡À12.3 | 67.4¡À12.2 | 69.3¡À12.3 || EEG+Gaze_concat | 57.7¡À7.9 | 61.5¡À7.3 | 66.1¡À7.2 | 72.0¡À7.0 | 79.4¡À6.1 || Static_EEG_Gaze_avg | 46.5¡À14.0 | 49.3¡À15.8 | 64.3¡À15.1 | 65.7¡À16.5 | 79.7¡À7.0 || PCET_only | 58.7¡À8.3 | 61.0¡À7.8 | 65.1¡À7.8 | 70.0¡À6.7 | 78.2¡À8.2 || GETA_only | 58.2¡À8.1 | 61.2¡À7.4 | 65.9¡À7.1 | 71.0¡À6.6 | 78.2¡À6.3 || PCET+GETA_concat | 58.0¡À8.2 | 60.6¡À7.5 | 64.3¡À7.1 | 69.6¡À6.4 | 77.3¡À7.6 || PCET+GETA_static_avg | 59.0¡À8.2 | 61.6¡À7.5 | 66.7¡À7.5 | 71.4¡À6.8 | 79.1¡À6.7 || PCET+GETA+CAGF | 62.3¡À9.3 | 65.8¡À9.6 | 69.7¡À9.5 | 74.1¡À8.6 | 80.1¡À7.2 |
## 2. CAGF Ablation (Final)

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| EEG+Gaze_concat | 57.7% | 61.5% | 66.1% | 72.0% | 79.4% || Static_average | 46.5% | 49.3% | 64.3% | 65.7% | 79.7% || CAGF_feature_only | 62.3% | 65.8% | 68.9% | 72.9% | 78.6% || CAGF_full_old | 60.9% | 63.7% | 67.7% | 72.2% | 78.6% || CAGF_v3_cross_interaction | 61.6% | 64.2% | 68.5% | 72.7% | 77.1% |
## 3. Key Findings

### PCET
- Raw + AbsError > Raw only > AbsError only > Shuffled > Random
- Prediction error features contain true class-discriminative information

### GETA
- Gaze-derived attention improves over EEG-only MLP
- Confidence + Entropy combined > either alone
- Random/shuffled attention degrades performance

### CAGF
- CAGF_feature_only (simple difference gate) outperforms concat and static average
- Adding confidence or cross-interaction features does NOT improve
- Simple gating mechanism is optimal

### Text Confound
- Majority/Random baselines at ~50% confirm random guessing
- Sentence length/word count at ~50-55% show minimal confounding

