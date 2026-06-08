# CNO-GSM Ablation Effect Audit

A module is marked effective only when its measured mean score exceeds the AdaGTCN-aligned baseline by the configured threshold. This file is an audit aid, not a substitute for full multi-seed statistical testing.

| Model | Module | Mean | Delta vs AdaGTCN-aligned | Effective |
|---|---|---:|---:|---|
| eeg_only_graph_tcn | EEG graph temporal baseline | 0.4044 | 0.0711 | yes |
| eeg_graph_ssm | Isolation: EEG graph with state-space temporal encoder | 0.6404 | 0.3070 | yes |
| gaze_only_ssm | Isolation: gaze-only state-space encoder | 0.6465 | 0.3131 | yes |
| adagtcn_aligned | AdaGTCN-aligned EEG+gaze temporal baseline | 0.3333 | 0.0000 | baseline |
| gaze_control_ssm | Module 1: gaze-controlled state-space temporal encoder | 0.6331 | 0.2997 | yes |
| bipartite_graph_ssm | Module 2: neuro-oculomotor bipartite graph | 0.6119 | 0.2785 | yes |
| bridge_bipartite_ssm | Module 3: subject-invariant bridge | 0.6215 | 0.2882 | yes |
| full_cnogsm | Modules 1-5: full CNO-GSM | 0.7168 | 0.3835 | yes |
