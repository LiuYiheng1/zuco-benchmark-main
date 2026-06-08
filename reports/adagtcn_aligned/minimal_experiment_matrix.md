# Minimal Experiment Matrix for CNO-GSM

Protocol name: AdaGTCN-aligned available-Y subject-independent protocol.

Data wording: word-level EEG band representations plus eye-tracking sequences. Do not describe the EEG input as raw EEG waveform sequence.

Goal: keep the experiment matrix small enough to be reproducible, but sufficient to decide whether the current CNO-GSM modules are real improvements over `adagtcn_aligned`.

## Current Code Anchors

Main training entry:

- `src/adagtcn_aligned/train_cnogsm.py`

Current model presets:

- `src/adagtcn_aligned/models.py`, `build_model`
- `eeg_only_graph_tcn`
- `adagtcn_aligned`
- `eeg_graph_ssm`
- `gaze_only_ssm`
- `gaze_control_ssm`
- `bipartite_graph_ssm`
- `bridge_bipartite_ssm`
- `full_cnogsm`

Current auxiliary loss weights:

- `--w-subject-adv`
- `--w-bridge-recon`
- `--w-common-align`
- `--w-unique-decor`
- `--w-graph-smooth`
- `--w-graph-entropy`

Formal runs must use:

- `--debug-random-split` disabled
- train-only normalization
- the fixed subject split in `reports/adagtcn_aligned/subject_splits.json`

## Minimal Run Policy

For screening:

- Run seed 0 first to identify obvious failures.
- Do not claim a module from seed 0 alone.

For paper-facing results:

- Run at least seeds 0, 1, and 2.
- Report mean and standard deviation.
- Use macro-F1 as the primary selection metric.

Recommended full-data input:

```bash
python -m src.adagtcn_aligned.train_cnogsm ^
  --sequence-jsonl data/adagtcn_aligned/paper_y16_full_band_vectors_sequences.jsonl ^
  --split-json reports/adagtcn_aligned/subject_splits.json ^
  --protocol Y16_12_2_2_seed0 ^
  --model MODEL_NAME ^
  --seed SEED ^
  --output-dir results/adagtcn_aligned/minimal_matrix
```

## 1. Main Comparison

These runs are the smallest formal comparison table. They answer whether the integrated CNO-GSM line is better than the AdaGTCN-aligned baseline, and whether the gains come from compact modules rather than the full stack only.

| ID | Model | Current availability | Purpose | Formal status |
|---|---|---|---|---|
| M1 | `eeg_only_graph_tcn` | Available in `train_cnogsm.py` and `models.py` | EEG-only graph TCN baseline | Required |
| M2 | `adagtcn_aligned` | Available in `train_cnogsm.py` and `models.py` | Primary AdaGTCN-aligned EEG+gaze baseline | Required |
| M3 | `gaze_concat_baseline` | Not exposed in `src/adagtcn_aligned.models.build_model`; only include if an already available simple concat baseline is adapted to the same JSONL, split, masks, and train-only normalization | Simple EEG+gaze concat control | Conditional required if available |
| M4 | `gaze_control_ssm` | Available | Main compact gaze-controlled state-space model | Required |
| M5 | `bipartite_graph_ssm` | Available | Adds EEG-gaze bipartite cross edges | Required |
| M6 | `bridge_bipartite_ssm` | Available | Adds subject bridge on top of bipartite graph | Required |
| M7 | `full_cnogsm` | Available | Full integrated CNO-GSM stack | Required |

Notes:

- `gaze_concat_baseline` must not be borrowed from another protocol unless it uses the exact same available-Y subject-independent split and train-only normalization.
- If `gaze_concat_baseline` is unavailable under this protocol, mark it as `N/A`, not as missing evidence for CNO-GSM.
- `adagtcn_aligned` must have finite training loss before it is used as the formal reference. If its history has `loss=nan`, rerun or fix the baseline before final claims.

## 2. Critical Ablations

These ablations are the minimum needed to decide which CNO-GSM modules can be kept. Do not add larger replacement models.

| ID | Ablation | Positive model | Negative/control model | Current implementation path | What it proves | Required metric result |
|---|---|---|---|---|---|---|
| A1 | w/o gaze control | `gaze_control_ssm` | `gaze_control_ssm_no_gaze_control` | Not currently exposed as a preset. Minimal switch only: keep EEG+gaze fusion unchanged, but pass zero gaze control into `GazeControlledStateSpace` | Whether gaze controls the state transition, not merely fusion input | Positive model must improve macro-F1 or balanced accuracy over the no-control version and over `adagtcn_aligned` |
| A2 | w/o EEG-gaze cross edges | `bipartite_graph_ssm` | `gaze_control_ssm` | Already available through existing presets | Whether the bipartite EEG-gaze graph adds useful cross-modal structure | `bipartite_graph_ssm` must improve macro-F1 or balanced accuracy over `gaze_control_ssm` and `adagtcn_aligned` |
| A3 | w/o subject bridge | `bridge_bipartite_ssm` | `bipartite_graph_ssm` | Already available through existing presets | Whether the subject bridge helps subject-independent generalization | `bridge_bipartite_ssm` must improve macro-F1 or balanced accuracy over `bipartite_graph_ssm` and `adagtcn_aligned` |
| A4 | w/o subject adversarial loss | `bridge_bipartite_ssm` default | `bridge_bipartite_ssm --w-subject-adv 0` | Already available through loss weight | Whether the adversarial subject objective helps beyond bridge reconstruction/capacity | Default must improve macro-F1 or balanced accuracy; subject probe should show reduced subject leakage |
| A5 | w/o graph smoothness | `bipartite_graph_ssm` default | `bipartite_graph_ssm --w-graph-smooth 0` | Already available through loss weight | Whether graph regularization helps rather than only learned adjacency capacity | Default must improve macro-F1 or balanced accuracy without hurting AUROC |
| A6 | w/o reconstruction loss | `bridge_bipartite_ssm` default | `bridge_bipartite_ssm --w-bridge-recon 0` | Already available through loss weight | Whether bridge reconstruction preserves useful modality information | Default must improve macro-F1 or balanced accuracy; no class collapse |

Optional but useful if `full_cnogsm` remains the reported best system:

| ID | Full-stack check | Positive model | Control |
|---|---|---|---|
| F1 | full w/o subject adversarial loss | `full_cnogsm` | `full_cnogsm --w-subject-adv 0` |
| F2 | full w/o graph smoothness | `full_cnogsm` | `full_cnogsm --w-graph-smooth 0` |
| F3 | full w/o reconstruction loss | `full_cnogsm` | `full_cnogsm --w-bridge-recon 0` |
| F4 | full w/o common-unique losses | `full_cnogsm` | `full_cnogsm --w-common-align 0 --w-unique-decor 0` |

These full-stack checks are not new models. They only zero existing loss weights.

## 3. Sanity Checks

Sanity checks are not formal performance results. They decide whether the formal results are credible.

| ID | Sanity check | Implementation | Expected result | Fail condition |
|---|---|---|---|---|
| S1 | random label | Shuffle labels within the training protocol and run at least `adagtcn_aligned` and the current best CNO-GSM model | Macro-F1, balanced accuracy, and AUROC should drop near chance/majority behavior | If a model remains close to real-label performance, suspect leakage or label artifacts |
| S2 | random subject split only for smoke test | Use `--debug-random-split` only to test code execution | Runs should complete; metrics are not publishable | Any random-split result included in formal tables invalidates the table |
| S3 | train-only normalization audit | Verify feature statistics are fit only on train subjects in `src/adagtcn_aligned/dataset.py`; compare saved meta with split subjects | No val/test subject should affect normalization statistics | If normalization uses all records or val/test records, rerun all formal experiments |
| S4 | subject classifier on learned representation | Train a simple linear/logistic subject probe on frozen pooled representations; compare no-bridge vs bridge models | Bridge/adversarial variants should reduce subject predictability without reducing class macro-F1 | If subject predictability stays high and class performance improves, bridge claim is unsupported |
| S5 | label distribution audit | Report class counts by train/val/test and by subject | Macro-F1 and balanced accuracy should explain performance beyond majority-class accuracy | If accuracy improves while macro-F1 or balanced accuracy drops, do not claim improvement |

## 4. Metrics

Primary metric:

- `test_macro_f1`

Required supporting metrics:

- `test_balanced_accuracy`
- `test_auroc`

Secondary metric only:

- `test_accuracy`

Reporting order:

1. Macro-F1
2. Balanced accuracy
3. AUROC
4. Accuracy

Accuracy must never be used as the sole basis for choosing a module because the available-Y classification labels can be imbalanced.

## 5. Decision Rule

Keep a module only if:

1. It improves macro-F1 or balanced accuracy over `adagtcn_aligned`.
2. It does not fail any sanity check.
3. Its immediate ablation comparison supports the mechanism.

Do not keep a module as a main claim if:

1. Accuracy improves but macro-F1 decreases.
2. Balanced accuracy decreases while accuracy increases.
3. The gain appears only under `--debug-random-split`.
4. The gain disappears under random-label or subject-leakage checks.
5. The improvement exists only inside `full_cnogsm` but not in the module's isolated ablation.

If `full_cnogsm` underperforms a compact ablation:

- Use the best compact ablation as the main model.
- Move the underperforming full stack to supplementary or remove it.
- Do not preserve the full CNO-GSM name by force.

## 6. Minimum Evidence Needed Per Claim

| Claim | Minimum evidence |
|---|---|
| CNO-GSM improves over AdaGTCN-aligned baseline | `full_cnogsm` or the selected compact main model beats `adagtcn_aligned` in macro-F1 or balanced accuracy across formal seeds |
| Gaze-controlled state-space dynamics are useful | A1 passes: `gaze_control_ssm` beats no-gaze-control SSM and `adagtcn_aligned` |
| EEG-gaze bipartite cross edges are useful | A2 passes: `bipartite_graph_ssm` beats `gaze_control_ssm` and `adagtcn_aligned` |
| Subject bridge is useful | A3 and A4 pass: bridge beats no-bridge, adversarial loss helps, subject probe leakage decreases |
| Graph smoothness is useful | A5 passes: default graph smoothness beats `--w-graph-smooth 0` |
| Reconstruction loss is useful | A6 passes: bridge reconstruction improves macro-F1 or balanced accuracy without class collapse |

## 7. Minimal Final Table Layout

Formal result table:

| Model | Macro-F1 | Balanced Acc. | AUROC | Accuracy | Keep/Drop |
|---|---:|---:|---:|---:|---|
| `eeg_only_graph_tcn` | mean ± std | mean ± std | mean ± std | mean ± std | baseline |
| `adagtcn_aligned` | mean ± std | mean ± std | mean ± std | mean ± std | primary baseline |
| `gaze_concat_baseline` | mean ± std or N/A | mean ± std or N/A | mean ± std or N/A | mean ± std or N/A | simple fusion baseline if available |
| `gaze_control_ssm` | mean ± std | mean ± std | mean ± std | mean ± std | decide by rule |
| `bipartite_graph_ssm` | mean ± std | mean ± std | mean ± std | mean ± std | decide by rule |
| `bridge_bipartite_ssm` | mean ± std | mean ± std | mean ± std | mean ± std | decide by rule |
| `full_cnogsm` | mean ± std | mean ± std | mean ± std | mean ± std | main only if best or clearly justified |

Ablation table:

| Module/loss | Full setting | Ablated setting | Delta macro-F1 | Delta BAcc | Decision |
|---|---|---|---:|---:|---|
| gaze control | `gaze_control_ssm` | no gaze control | value | value | Keep/Drop |
| EEG-gaze cross edges | `bipartite_graph_ssm` | `gaze_control_ssm` | value | value | Keep/Drop |
| subject bridge | `bridge_bipartite_ssm` | `bipartite_graph_ssm` | value | value | Keep/Drop |
| subject adversarial loss | default | `--w-subject-adv 0` | value | value | Keep/Drop |
| graph smoothness | default | `--w-graph-smooth 0` | value | value | Keep/Drop |
| reconstruction loss | default | `--w-bridge-recon 0` | value | value | Keep/Drop |

## 8. What Not To Add

Do not add these to the minimal matrix:

- EEGPT-style pretraining
- causal invariant factor models
- new transformer/foundation-model encoders
- additional large multimodal fusion modules
- extra losses without a direct ablation
- random subject split as a formal result

The current priority is not a larger model. The priority is proving whether the existing CNO-GSM modules survive strict ablation under the AdaGTCN-aligned available-Y subject-independent protocol.
