# CNO-GSM Ablation Effect Audit

A module is marked effective only when its measured mean score exceeds the AdaGTCN-aligned baseline by the configured threshold. This file is an audit aid, not a substitute for full multi-seed statistical testing.

| Model | Module | Mean | Delta vs AdaGTCN-aligned | Effective |
|---|---|---:|---:|---|
| eeg_only_graph_tcn | EEG graph temporal baseline | 0.5756 | -0.0068 | no |
| eeg_graph_ssm | Isolation: EEG graph with state-space temporal encoder | 0.5687 | -0.0137 | no |
| gaze_only_ssm | Isolation: gaze-only state-space encoder | 0.5955 | 0.0131 | yes |
| adagtcn_aligned | AdaGTCN-aligned EEG+gaze temporal baseline | 0.5824 | 0.0000 | baseline |
| gaze_control_ssm | Module 1: gaze-controlled state-space temporal encoder | 0.5952 | 0.0128 | yes |
| bipartite_graph_ssm | Module 2: neuro-oculomotor bipartite graph | 0.5444 | -0.0380 | no |
| bridge_bipartite_ssm | Module 3: subject-invariant bridge | 0.6246 | 0.0422 | yes |
| full_cnogsm | Modules 1-5: full CNO-GSM | 0.6009 | 0.0185 | yes |
