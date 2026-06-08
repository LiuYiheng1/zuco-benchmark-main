# Comprehensive Experiment Report

## 1. Main Comparison with Zuco Baselines

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| EEG_SVM | 57.8% | 60.9% | 64.5% | 69.1% | 75.6% || Gaze_SVM | 59.6% | 62.1% | 64.8% | 67.8% | 69.3% || EEG_Gaze_SVM | 42.8% | 43.9% | 59.1% | 65.3% | 76.9% || EEG_PCA_SVM | nan% | nan% | nan% | nan% | 75.0% || EEG_LSTM_proxy | nan% | nan% | nan% | nan% | 78.2% || Gaze_LSTM_proxy | nan% | nan% | nan% | nan% | 68.1% || EEG_GCN_proxy | nan% | nan% | nan% | nan% | 78.2% || EEG_Gaze_LSTM_proxy | nan% | nan% | nan% | nan% | 78.5% |
## 2. Text Confound Controls

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| Majority | 46.1% | 46.1% | 46.1% | 46.1% | 46.1% || Random | 50.6% | 50.6% | 50.6% | 50.6% | 50.6% || SentenceLength | 53.9% | 53.9% | 53.9% | 53.9% | 53.9% || WordCount | 53.9% | 53.9% | 53.9% | 53.9% | 53.9% |
## 3. GETA Ablation

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| EEG_MLP | nan% | nan% | nan% | nan% | 78.2% || Gaze_MLP | nan% | nan% | nan% | nan% | 69.0% || GETA_confidence | nan% | nan% | nan% | nan% | 78.0% || GETA_entropy | nan% | nan% | nan% | nan% | 75.8% || GETA_conf_ent | nan% | nan% | nan% | nan% | 78.0% || GETA_random_att | nan% | nan% | nan% | nan% | 75.5% || GETA_shuffled_att | - | - | - | - | - |
## 4. PCET Ablation

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| PCET_raw | nan% | nan% | nan% | nan% | 77.9% || PCET_abserror | nan% | nan% | nan% | nan% | 73.1% || PCET_raw_abserror | nan% | nan% | nan% | nan% | 77.9% || PCET_shuffled | nan% | nan% | nan% | nan% | 77.7% || PCET_random | nan% | nan% | nan% | nan% | 77.8% |
## 5. Gaze Feature Baselines

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| Gaze_SVM | 59.6% | 62.1% | 64.8% | 67.8% | 69.3% || Gaze_Fixation_SVM | nan% | nan% | nan% | nan% | 66.5% || Gaze_Saccade_SVM | nan% | nan% | nan% | nan% | 68.2% || Gaze_FixSacc_SVM | nan% | nan% | nan% | nan% | 69.3% |