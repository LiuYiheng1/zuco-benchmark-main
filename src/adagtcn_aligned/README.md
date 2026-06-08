# AdaGTCN-Aligned Track

This folder is the clean alignment layer for AdaGTCN-style work. It is separate
from historical `AdaGTCN-lite` and sentence-level proxy experiments.

## Alignment Rules

1. Protocol alignment: use subject-independent splits only.
2. Available-Y primary split: `Y16_12_2_2_seed0`.
3. Stability split: `Y16_LOSO_*`.
4. Do not call the available-Y split `12/2/4`; this workspace has 16 labeled Y
   subjects, not the 18 needed for 12 train / 2 validation / 4 test.
5. Input alignment: use word/fixation-level EEG and eye-tracking sequences from
   raw `.mat` files, not only sentence-level feature vectors.
6. Metrics alignment: report accuracy and macro-F1, with the AdaGTCN literature
   table kept as external reference unless the exact 18-subject split is
   recovered.

## Commands

Create split manifests:

```bash
python -m src.adagtcn_aligned.protocol --data-dir data/train --output-dir reports/adagtcn_aligned --seed 0
```

Smoke-test sequence extraction without EEG:

```bash
python -m src.adagtcn_aligned.extract_word_sequences --data-dir data/train --output-dir data/adagtcn_aligned --subjects YAC --max-sentences 1 --eeg-mode none
```

Smoke-test sequence extraction with one EEG band field:

```bash
python -m src.adagtcn_aligned.extract_word_sequences --data-dir data/train --output-dir data/adagtcn_aligned --subjects YAC --max-sentences 1 --max-words 1 --eeg-mode band_means --eeg-fields TRT_t1
```

Create an AdaGTCN-aligned word sequence file with 105-electrode band vectors:

```bash
python -m src.adagtcn_aligned.extract_word_sequences --data-dir data/train --output-dir data/adagtcn_aligned --prefix y16_word_band_vectors --eeg-mode band_vectors --eeg-fields TRT_t1,TRT_t2,TRT_a1,TRT_a2,TRT_b1,TRT_b2,TRT_g1,TRT_g2
```

Run comparable ablations on the available-Y subject split:

```bash
python -m src.adagtcn_aligned.train_cnogsm --sequence-jsonl data/adagtcn_aligned/y16_word_band_vectors_sequences.jsonl --protocol Y16_12_2_2_seed0 --model all --epochs 30 --batch-size 32 --hidden-dim 64 --max-len 80 --output-dir results/adagtcn_aligned
```

Audit which modules are actually effective:

```bash
python -m src.adagtcn_aligned.analyze_ablation_effects --inputs results/adagtcn_aligned/cnogsm_all_seed0.csv --metric test_macro_f1 --output-dir reports/adagtcn_aligned
```

Important: `--debug-random-split` is only for smoke tests. Do not report those
numbers as AdaGTCN-aligned results.

Isolation ablations are included to check whether gains come from EEG, gaze, or
their interaction:

- `eeg_graph_ssm`: EEG graph plus state-space temporal encoder, with gaze zeroed.
- `gaze_only_ssm`: gaze-controlled state-space encoder, with EEG zeroed.
- `gaze_control_ssm`: EEG and gaze together without the bipartite graph.
