# Formal CNO-GSM Experiment Report: BCI/CUDA Seed 0

Protocol: AdaGTCN-aligned available-Y subject-independent protocol.

This is not an original AdaGTCN 12/2/4 reproduction. The local formal split uses 16 labeled Y subjects.

Input wording: word-level EEG band representations from `TRT_*` band vectors plus eye-tracking sequences. These are not raw EEG waveform sequences.

## Execution Summary

Formal run:

```text
D:\Anaconda\envs\BCI\python.exe -m src.adagtcn_aligned.train_cnogsm
  --sequence-jsonl data/adagtcn_aligned/paper_y16_full_band_vectors_sequences.jsonl
  --split-json reports/adagtcn_aligned/subject_splits.json
  --protocol Y16_12_2_2_seed0
  --model all
  --seed 0
  --device cuda
  --output-dir results/adagtcn_aligned/paper_y16_bci_cuda
```

Environment:

| Item | Value |
|---|---|
| Python | `D:\Anaconda\envs\BCI\python.exe`, Python 3.10.8 |
| Torch | 2.3.0+cu118 |
| CUDA visible to torch | True |
| GPU | NVIDIA GeForce RTX 3090 |
| NumPy | 1.26.4 |
| scikit-learn | 1.7.2 |

Output files:

| Artifact | Path |
|---|---|
| Result CSV | `results/adagtcn_aligned/paper_y16_bci_cuda/cnogsm_all_seed0.csv` |
| Run meta | `results/adagtcn_aligned/paper_y16_bci_cuda/cnogsm_all_seed0_meta.json` |
| Stdout log | `results/adagtcn_aligned/paper_y16_bci_cuda/run_all_seed0_bci_cuda.out.log` |
| Stderr log | `results/adagtcn_aligned/paper_y16_bci_cuda/run_all_seed0_bci_cuda.err.log` |
| Ablation audit | `reports/adagtcn_aligned/paper_y16_bci_cuda/cnogsm_ablation_effect_audit.csv` |

## Protocol Checks

| Check | Status | Evidence |
|---|---|---|
| Formal split used | Pass | `protocol=Y16_12_2_2_seed0` |
| Debug random split disabled | Pass | All result rows have `debug_random_split=False` |
| Subject overlap | Pass | Train/val/test subject intersections are empty |
| Data size | Pass | train=8746, val=1260, test=1478 |
| EEG input wording | Pass | `eeg_dim=840`, from 8 `TRT_*` bands x 105 electrodes |
| Training loss finite | Pass | All 8 history files have finite loss values |
| Train-only normalization | Code path pass | `src/adagtcn_aligned/dataset.py` fits stats using `fit_feature_stats(..., train_examples, ...)` |
| Padding mask | Code path pass | sequence pooling uses `step_mask` |
| Missing fixation mask | Needs further sanity check | Main fusion path still encodes zero-filled missing gaze positions before temporal pooling; do not make final paper claims before a dedicated mask audit |

Subject split:

| Split | Subjects |
|---|---|
| Train | YMD, YSL, YFR, YAG, YLS, YAK, YDG, YRK, YSD, YHS, YIS, YDR |
| Val | YAC, YFS |
| Test | YTL, YRP |

Label distribution:

| Split | Label 0 | Label 1 | Total |
|---|---:|---:|---:|
| Train | 4608 | 4138 | 8746 |
| Val | 661 | 599 | 1260 |
| Test | 780 | 698 | 1478 |

## Main Result Table

Primary metric: `test_macro_f1`.

Accuracy is secondary only.

| Rank | Model | test_macro_f1 | test_balanced_accuracy | test_auroc | test_accuracy | Delta macro-F1 vs `adagtcn_aligned` | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `bridge_bipartite_ssm` | 0.6246 | 0.6395 | 0.6565 | 0.6306 | +0.0422 | Best compact ablation; use as current main model candidate |
| 2 | `full_cnogsm` | 0.6009 | 0.6460 | 0.7857 | 0.6292 | +0.0185 | Improves baseline but underperforms compact ablation on macro-F1 |
| 3 | `gaze_only_ssm` | 0.5955 | 0.6344 | 0.7043 | 0.6191 | +0.0131 | Strong gaze-only baseline; not a CNO-GSM module claim |
| 4 | `gaze_control_ssm` | 0.5952 | 0.6422 | 0.7810 | 0.6252 | +0.0128 | Passes vs baseline, but not better than gaze-only on macro-F1 |
| 5 | `adagtcn_aligned` | 0.5824 | 0.6299 | 0.7741 | 0.6130 | 0.0000 | Primary baseline |
| 6 | `eeg_only_graph_tcn` | 0.5756 | 0.6180 | 0.5824 | 0.6022 | -0.0068 | Fails vs baseline |
| 7 | `eeg_graph_ssm` | 0.5687 | 0.5906 | 0.5708 | 0.5798 | -0.0137 | Fails vs baseline |
| 8 | `bipartite_graph_ssm` | 0.5444 | 0.5476 | 0.5475 | 0.5447 | -0.0380 | Fails; do not claim bipartite graph alone |

## Main Finding

The best seed0 model under the BCI/CUDA formal run is:

```text
bridge_bipartite_ssm
```

Because `full_cnogsm` is not the best model by the primary metric, the current main model should be the best compact ablation, `bridge_bipartite_ssm`, not `full_cnogsm`.

This follows the project rule:

> If the full model underperforms a compact ablation, use the compact ablation as the main model.

## Module Decisions From This Run

| Module / comparison | Evidence | Decision |
|---|---|---|
| Gaze-controlled state-space dynamics | `gaze_control_ssm` improves macro-F1 over `adagtcn_aligned` by +0.0128 and BAcc by +0.0123 | Keep as a candidate, but claim cautiously |
| Gaze-only SSM baseline | `gaze_only_ssm` is slightly above `gaze_control_ssm` in macro-F1 | Must be treated as a strong control; do not claim multimodal fusion clearly beats gaze-only |
| EEG-gaze bipartite graph alone | `bipartite_graph_ssm` drops below `gaze_control_ssm` by -0.0508 macro-F1 and below `adagtcn_aligned` by -0.0380 | Failed as an isolated module |
| Subject bridge on bipartite model | `bridge_bipartite_ssm` improves over `bipartite_graph_ssm` by +0.0802 macro-F1 and over `adagtcn_aligned` by +0.0422 | Strongest current compact candidate |
| Full CNO-GSM stack | `full_cnogsm` improves over `adagtcn_aligned` but is below `bridge_bipartite_ssm` by -0.0238 macro-F1 | Do not use full stack as main model |
| Common-unique/full extra losses | Only present in `full_cnogsm`; full is worse than `bridge_bipartite_ssm` on macro-F1 | Do not claim as main contribution from this evidence |

## Interpretation

The current evidence supports a compact model line rather than the full CNO-GSM stack:

```text
gaze_control_ssm -> bridge_bipartite_ssm
```

The strongest claim that can be made from this seed0 run is:

> Under the AdaGTCN-aligned available-Y subject-independent protocol, a compact bridge-regularized EEG-gaze dynamical model outperforms the local `adagtcn_aligned` baseline on macro-F1.

The following claims are not supported:

- The full CNO-GSM stack is the best model.
- The bipartite EEG-gaze graph alone is effective.
- EEG-only state-space modeling improves over `adagtcn_aligned`.
- Multimodal gaze control is clearly better than gaze-only dynamics.
- Any result is a reproduction of original AdaGTCN 12/2/4.
- Any result uses raw EEG waveform modeling.

## Limitations Before Paper Claims

This is a formal seed0 run, but it is still not enough for final paper claims.

Required before writing paper-level claims:

1. Run at least seeds 1 and 2 under the same BCI/CUDA environment.
2. Run the missing sanity checks:
   - random label
   - train-only normalization audit artifact
   - subject classifier on learned representation
   - label distribution audit in the final report table
   - missing fixation / mask audit
3. Add or verify the critical loss ablations:
   - `bridge_bipartite_ssm --w-subject-adv 0`
   - `bridge_bipartite_ssm --w-bridge-recon 0`
   - `bridge_bipartite_ssm --w-graph-smooth 0`
4. Do not use `full_cnogsm` as the paper main model unless later seeds reverse the macro-F1 ranking.

## Current Recommendation

Use `bridge_bipartite_ssm` as the current main model candidate.

Move `full_cnogsm` to supplementary unless later multi-seed results show it is better by macro-F1.

Mark `bipartite_graph_ssm` alone as failed in this run.
