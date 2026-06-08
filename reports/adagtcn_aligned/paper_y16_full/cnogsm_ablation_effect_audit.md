# CNO-GSM Ablation Effect Audit

A module is marked effective only when its measured mean score exceeds the AdaGTCN-aligned baseline by the configured threshold. This file is an audit aid, not a substitute for full multi-seed statistical testing.

| Model | Module | Mean | Delta vs AdaGTCN-aligned | Effective |
|---|---|---:|---:|---|
| eeg_graph_ssm | Isolation: EEG graph with state-space temporal encoder | 0.5952 | 0.2498 | yes |
| gaze_only_ssm | Isolation: gaze-only state-space encoder | 0.6361 | 0.2907 | yes |
| adagtcn_aligned | AdaGTCN-aligned EEG+gaze temporal baseline | 0.3454 | 0.0000 | baseline |
| gaze_control_ssm | Module 1: gaze-controlled state-space temporal encoder | 0.5764 | 0.2310 | yes |
| full_cnogsm | Modules 1-5: full CNO-GSM | 0.6544 | 0.3090 | yes |
