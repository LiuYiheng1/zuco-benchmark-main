# Pilot Conclusion: CNO-GSM vs AdaGTCN-Aligned Baseline

## Setting

- Data: 16 labeled Y subjects from official ZuCo raw Matlab files.
- Protocol: `Y16_12_2_2_seed0`, subject-independent.
- Pilot subset: first 60 sentences per task/file where available.
- Input: word/fixation-level EEG and eye-tracking sequences.
- EEG representation: 105-electrode band vectors from `TRT_t1,TRT_t2,TRT_a1,TRT_a2,TRT_b1,TRT_b2,TRT_g1,TRT_g2`.
- Train/val/test examples: 1415 / 240 / 240.
- Metric emphasized: test macro-F1.

This is a real subject-split pilot, not a debug random split. It is still not
the final full-data experiment.

## Results

| Model | Test Acc | Test Macro-F1 | Test AUROC | Seed notes |
|---|---:|---:|---:|---|
| `adagtcn_aligned` | 0.5000 | 0.3333 | 0.4159 / 0.5303 | seed0 and seed1 both majority-like |
| `eeg_only_graph_tcn` | 0.5250 | 0.4044 | 0.4966 | seed0 |
| `eeg_graph_ssm` | 0.6542 | 0.6404 | 0.8002 | seed0 |
| `gaze_only_ssm` | 0.6708 | 0.6465 | 0.7550 | seed0 |
| `gaze_control_ssm` | 0.6708 | 0.6331 | 0.7931 | seed0 |
| `bipartite_graph_ssm` | 0.6458 | 0.6119 | 0.6526 | seed0 |
| `bridge_bipartite_ssm` | 0.6625 | 0.6215 | 0.8519 | seed0 |
| `full_cnogsm` | 0.7333 / 0.6500 | 0.7168 / 0.6267 | 0.7058 / 0.7566 | seed0 and seed1 |

Mean over available runs:

- `adagtcn_aligned`: macro-F1 0.3333.
- `full_cnogsm`: macro-F1 0.6717.
- Absolute gain: +0.3384 macro-F1.

## What Looks Useful

1. State-space temporal modeling is useful.
   - `eeg_only_graph_tcn`: 0.4044 macro-F1.
   - `eeg_graph_ssm`: 0.6404 macro-F1.
   - The main improvement is not just fusion; replacing TCN with a state-space
     temporal encoder is a large gain.

2. Gaze is useful.
   - `gaze_only_ssm`: 0.6465 macro-F1.
   - Eye movement contains strong task information.

3. EEG is also useful.
   - `eeg_graph_ssm`: 0.6404 macro-F1, close to gaze-only.
   - The result is not purely an eye-tracking shortcut.

4. Full multimodal interaction is currently best.
   - `full_cnogsm`: 0.7168 on seed0, 0.6267 on seed1.
   - The two-seed mean remains above both `eeg_graph_ssm` and `gaze_only_ssm`.

5. The subject bridge is promising but needs tuning.
   - `bridge_bipartite_ssm` improves slightly over `bipartite_graph_ssm` in
     macro-F1 and strongly in AUROC.
   - It should be kept for now, but the reconstruction/adversarial weights need
     a grid search.

## What Needs Caution

1. The local `adagtcn_aligned` baseline is not the original AdaGTCN paper model.
   It is an aligned local baseline with graph + gaze + TCN. A stronger
   reproduction baseline should still be tuned before paper claims.

2. The pilot uses only the first 60 sentences per task/file, not all data.
   The next required step is full extraction or multiple random sentence
   subsets.

3. Module effectiveness is based on one seed for most ablations and two seeds
   only for `adagtcn_aligned` and `full_cnogsm`.

## Immediate Next Changes

1. Keep `full_cnogsm`, `eeg_graph_ssm`, `gaze_only_ssm`, and `gaze_control_ssm`.
2. Retune `bipartite_graph_ssm` and `bridge_bipartite_ssm` rather than deleting
   them.
3. Run at least three seeds for `adagtcn_aligned`, `eeg_graph_ssm`,
   `gaze_only_ssm`, and `full_cnogsm`.
4. Run a full-data or larger-subset experiment after confirming runtime.

