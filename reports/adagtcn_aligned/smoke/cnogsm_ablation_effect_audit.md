# CNO-GSM Ablation Effect Audit

A module is marked effective only when its measured mean score exceeds the AdaGTCN-aligned baseline by the configured threshold. This file is an audit aid, not a substitute for full multi-seed statistical testing.

| Model | Module | Mean | Delta vs AdaGTCN-aligned | Effective |
|---|---|---:|---:|---|
| eeg_only_graph_tcn | EEG graph temporal baseline | 0.5000 | 0.0714 | yes |
| adagtcn_aligned | AdaGTCN-aligned EEG+gaze temporal baseline | 0.4286 | 0.0000 | baseline |
| gaze_control_ssm | Module 1: gaze-controlled state-space temporal encoder | 0.2000 | -0.2286 | no |
| bipartite_graph_ssm | Module 2: neuro-oculomotor bipartite graph | 0.5000 | 0.0714 | yes |
| bridge_bipartite_ssm | Module 3: subject-invariant bridge | 0.2000 | -0.2286 | no |
| full_cnogsm | Modules 1-5: full CNO-GSM | 0.4286 | 0.0000 | no |
