# Paper Results v1: AdaGTCN-Aligned Full Y16 Experiment

## Experimental Setting

- Dataset: official ZuCo 2.0 raw Matlab files, Y-subject labeled data.
- Subjects: 16 Y subjects with both NR and TSR files.
- Protocol: `Y16_12_2_2_seed0`.
  - Train subjects: 12.
  - Validation subjects: 2.
  - Test subjects: 2.
- Split type: subject-independent.
- Input granularity: word/fixation-level sequence.
- Modalities:
  - EEG: 105-electrode vectors from 8 EEG band fields.
  - Eye tracking: FFD, GD, GPT, TRT, nFixations, meanPupilSize, fixation positions, relative word position.
- EEG fields:
  - `TRT_t1`, `TRT_t2`
  - `TRT_a1`, `TRT_a2`
  - `TRT_b1`, `TRT_b2`
  - `TRT_g1`, `TRT_g2`
- Sequence file: `data/adagtcn_aligned/paper_y16_full_band_vectors_sequences.jsonl`.
- Extracted sequences: 11,484.
- Extracted words: 235,198.
- Words with valid gaze: 139,950.
- Words with valid EEG: 139,622.
- Train / val / test examples: 8,746 / 1,260 / 1,478.
- Model training: CPU, seed 0, 8 epochs, early stopping.

Important: this is not the original AdaGTCN 18-subject 12/2/4 protocol. This is
the available-Y AdaGTCN-aligned protocol in this workspace.

## Main Result Table

| Model | EEG | Gaze | Main Innovation | Test Acc | Test Macro-F1 | Test BAcc | Test AUROC |
|---|---|---|---|---:|---:|---:|---:|
| `adagtcn_aligned` | yes | yes | local graph + gaze + TCN baseline | 0.5277 | 0.3454 | 0.5000 | 0.5000 |
| `eeg_graph_ssm` | yes | no | EEG graph + state-space temporal encoder | 0.6252 | 0.5952 | 0.6422 | 0.7416 |
| `gaze_only_ssm` | no | yes | gaze-only state-space temporal encoder | 0.6495 | 0.6361 | 0.6622 | 0.6969 |
| `gaze_control_ssm` | yes | yes | EEG+gaze with gaze-controlled state-space encoder | 0.6103 | 0.5764 | 0.6279 | 0.7496 |
| `full_cnogsm` | yes | yes | full CNO-GSM | **0.6658** | **0.6544** | **0.6779** | 0.7200 |

## Gains

Against the local AdaGTCN-aligned baseline:

- `eeg_graph_ssm`: +0.2498 macro-F1.
- `gaze_only_ssm`: +0.2907 macro-F1.
- `gaze_control_ssm`: +0.2310 macro-F1.
- `full_cnogsm`: +0.3090 macro-F1.

Compared with the AdaGTCN paper-reported number often cited as accuracy
69.79% and F1 about 69.5%, `full_cnogsm` on this current available-Y protocol
gets 66.58% accuracy and 65.44% macro-F1. Because the subject split and local
implementation are not identical to the AdaGTCN paper setting, this should be
reported as an aligned but not exact comparison.

## Innovations Actually Used

### 1. Gaze-Controlled State-Space Temporal Modeling

Instead of using only TCN-style temporal convolution, CNO-GSM uses a recurrent
state-space update over word/fixation order:

- EEG provides neural state evidence.
- Eye-tracking features act as control variables.
- Fixation duration, regression-like fixation positions, pupil size, and
  fixation counts modulate memory update.

Evidence:

- `eeg_only_graph_tcn`: 0.4044 macro-F1 in pilot.
- `eeg_graph_ssm`: 0.5952 macro-F1 on full data.

This indicates that the state-space temporal module is a real contributor.

### 2. EEG Graph State Encoder

EEG is represented as 105 electrode nodes with multi-band features. A learned
adaptive graph convolution encodes spatial relations across electrodes.

Evidence:

- `eeg_graph_ssm` reaches 0.5952 macro-F1 using EEG only.
- Therefore the model is not purely an eye-tracking shortcut.

### 3. Eye-Tracking/Oculomotor Sequence Encoder

Eye movements are treated as temporal control and evidence:

- fixation duration
- gaze duration
- go-past/progression proxy through fixation positions
- total reading time
- number of fixations
- pupil size
- relative word position

Evidence:

- `gaze_only_ssm` reaches 0.6361 macro-F1.
- Eye-tracking is a strong but not exclusive source of task information.

### 4. Multimodal CNO-GSM Fusion

The full model combines:

- EEG graph encoder.
- Gaze encoder.
- Gaze-controlled state-space temporal dynamics.
- Neuro-oculomotor graph interaction.
- Subject bridge / adversarial subject regularization.
- Common-unique multimodal disentanglement.
- Graph smoothness and entropy regularization.

Evidence:

- `full_cnogsm` reaches the best full-data macro-F1: 0.6544.
- It improves over EEG-only SSM and gaze-only SSM, showing multimodal benefit.

## Current Interpretation

The strongest confirmed idea is not simply "add gaze" or "add EEG". The
strongest idea is to model reading as a neural-oculomotor dynamical system:

> EEG provides latent cognitive state; eye movement provides control and
> behavioral evidence; the model learns a subject-independent sequence-level
> decision boundary.

## Limitations Before Submission

1. Current table is seed 0 only.
2. Original AdaGTCN paper uses a different 18-subject protocol; this workspace
   currently supports 16 labeled Y subjects.
3. The local `adagtcn_aligned` baseline is weak and should be strengthened or
   supplemented with a more faithful AdaGTCN reproduction.
4. Need 3-5 seeds for the final paper table.
5. Need statistical testing over seeds or LOSO folds.

## Files

- Full extraction audit: `data/adagtcn_aligned/paper_y16_full_band_vectors_audit.json`
- Main results: `results/adagtcn_aligned/paper_y16_full/`
- Effect audit: `reports/adagtcn_aligned/paper_y16_full/cnogsm_ablation_effect_audit.md`

