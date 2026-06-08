# Research Innovation Audit

Audit date: 2026-06-04  
Project: `D:\pycharmproject\zuco-benchmark-main`  
Scope: `src/adagtcn_aligned`, `data/train`, `data/adagtcn_aligned`, `results/adagtcn_aligned`, `reports/adagtcn_aligned`

## Executive Verdict

The current project supports an **AdaGTCN-aligned available-Y subject-independent protocol**, not an original AdaGTCN 18-subject 12/2/4 reproduction. The strongest currently verifiable idea is **state-space temporal modeling over word-level EEG band representations and eye-tracking sequences**. The full `full_cnogsm` result is promising, but several claimed modules are entangled and under-ablated.

The most important audit risk is that the current `adagtcn_aligned` full-data baseline is weak and numerically unstable: `results/adagtcn_aligned/paper_y16_full/adagtcn_aligned_history_seed0.csv` records `loss=nan` for all logged epochs, and the test metrics are majority-like (`accuracy=0.5277`, `balanced_accuracy=0.5000`, `macro_f1=0.3454`, `auroc=0.5000`). Therefore, current gains over `adagtcn_aligned` should not yet be framed as gains over a faithful or strong AdaGTCN reproduction.

## Path Audit

Requested paths:

- `src/adagtcn_aligned/extract_word_sequences.py`: exists.
- `src/adagtcn_aligned/dataset.py`: exists.
- `src/adagtcn_aligned/models.py`: exists.
- `src/adagtcn_aligned/train_cnogsm.py`: exists.
- `src/adagtcn_aligned/analyze_ablation_effects.py`: exists.
- `src/adagtcn_aligned/subject_splits.json`: **missing**.
- `src/adagtcn_aligned/alignment_protocol_note.md`: **missing**.
- Actual split/protocol files are under `reports/adagtcn_aligned/subject_splits.json` and `reports/adagtcn_aligned/alignment_protocol_note.md`.
- `data/train`: exists with 32 `.mat` files.
- `data/adagtcn_aligned`: exists with smoke, pilot, and full sequence artifacts.
- `results/adagtcn_aligned`: exists with smoke, pilot, and full result CSV files.
- `reports/adagtcn_aligned`: exists with protocol notes, effect audits, pilot reports, and full-data report.

This path mismatch should be fixed in documentation or scripts before publication, but it is not itself a data leakage issue.

## 1. Does The Data Support Word/Fixation-Level EEG + Gaze Sequence Modeling?

**Verdict: yes, with precise wording.** The data supports **word-level / fixation-derived EEG band representations plus eye-tracking sequence modeling**. It should not be described as raw EEG waveform sequence modeling.

Evidence:

- Raw official Y-subject data exists in `data/train`: 32 `.mat` files, corresponding to 16 Y subjects with NR and TSR files.
- The extraction script maps task labels with `LABEL_TO_Y = {"NR": 1, "TSR": 0}` in `src/adagtcn_aligned/extract_word_sequences.py:25`.
- EEG fields are band-level features: `DEFAULT_EEG_FIELDS = ["TRT_t1", "TRT_t2", "TRT_a1", "TRT_a2", "TRT_b1", "TRT_b2", "TRT_g1", "TRT_g2"]` in `src/adagtcn_aligned/extract_word_sequences.py:29`.
- EEG extraction is performed in `read_eeg_features()` at `src/adagtcn_aligned/extract_word_sequences.py:96`; `band_vectors` appends up to 105 electrode values per field at `src/adagtcn_aligned/extract_word_sequences.py:122`.
- Gaze and EEG masks are serialized into each sequence record at `src/adagtcn_aligned/extract_word_sequences.py:240` and `src/adagtcn_aligned/extract_word_sequences.py:244`.

Full extraction artifact:

- `data/adagtcn_aligned/paper_y16_full_band_vectors_sequences.jsonl`
- `data/adagtcn_aligned/paper_y16_full_band_vectors_manifest.csv`
- `data/adagtcn_aligned/paper_y16_full_band_vectors_audit.json`

Full-data audit:

- Files: 32.
- Sequences: 11,484.
- Words: 235,198.
- Words with valid gaze: 139,950.
- Words with valid EEG: 139,622.
- EEG mode: `band_vectors`.
- EEG dimensionality in training metadata: 840 = 105 electrodes x 8 band fields.

Research wording that is currently defensible:

> word-level EEG band representations and eye-tracking sequences.

Research wording that is not defensible:

> raw EEG waveform sequence.

## 2. Does The Current Split Have Subject Overlap?

**Verdict: no subject overlap was found for the current full-data protocol.**

The actual protocol file is `reports/adagtcn_aligned/subject_splits.json`. The split generation logic is in `src/adagtcn_aligned/protocol.py:85`, where `make_y16_12_2_2()` creates 12 train, 2 validation, and 2 test subjects from the available 16 Y subjects.

Current `Y16_12_2_2_seed0` split:

- Train subjects: `YAG`, `YAK`, `YDG`, `YDR`, `YFR`, `YHS`, `YIS`, `YLS`, `YMD`, `YRK`, `YSD`, `YSL`.
- Validation subjects: `YAC`, `YFS`.
- Test subjects: `YRP`, `YTL`.
- Train/validation overlap: none.
- Train/test overlap: none.
- Validation/test overlap: none.

Label and sequence counts from `paper_y16_full_band_vectors_manifest.csv`:

| Split | NR label 1 | TSR label 0 | Total | Majority Acc |
|---|---:|---:|---:|---:|
| Train | 4,138 | 4,608 | 8,746 | 0.5269 |
| Validation | 599 | 661 | 1,260 | 0.5246 |
| Test | 698 | 780 | 1,478 | 0.5277 |

The labels are mildly imbalanced, and majority-class accuracy is around 0.52-0.53. Therefore, balanced accuracy and macro-F1 should be primary metrics, not raw accuracy alone.

## 3. Is Normalization Fit Only On Train Split?

**Verdict: yes, for the current `make_datasets()` path.**

Evidence:

- `collect_offsets()` filters examples by subject list in `src/adagtcn_aligned/dataset.py:145`.
- `fit_feature_stats()` is defined in `src/adagtcn_aligned/dataset.py:162`.
- In the normal non-debug branch, train, validation, and test examples are collected separately in `src/adagtcn_aligned/dataset.py:285` to `src/adagtcn_aligned/dataset.py:287`.
- Feature statistics are fitted only using `train_examples` at `src/adagtcn_aligned/dataset.py:291`.
- The same train-fitted stats are then passed into train, validation, and test dataset objects.

Caveat:

- `infer_dims()` scans all train+val+test offsets to infer EEG/gaze dimensionality. This is not a statistical normalization leak, but it is still a metadata pass over all splits. It should be acceptable because dimensions are fixed by feature design, but for stricter reproducibility the dimensions could be derived from configuration instead.
- `debug_random_split=True` intentionally randomizes examples and can introduce subject overlap. Formal result CSVs inspected here have `debug_random_split=False`.

## 4. Mask Audit

### 4.1 Padding Mask

**Verdict: mostly correct.**

Evidence:

- In `ZuCoSequenceDataset.__getitem__`, `step_mask` is initialized to zeros in `src/adagtcn_aligned/dataset.py:227`.
- For each real word up to `max_len`, `step_mask[idx] = 1.0` in `src/adagtcn_aligned/dataset.py:235`.
- Sequence pooling uses `masked_mean(seq_h, step_mask)` in `src/adagtcn_aligned/models.py:324`.

Current full data has no sequence longer than `max_len=80` in the manifest audit (`long80_rate=0.0` in train/val/test), so truncation is not currently a practical concern.

### 4.2 EEG Missing Mask

**Verdict: mostly correct, but missingness may still become a latent cue.**

Evidence:

- `eeg_masks` are read from JSONL in `src/adagtcn_aligned/dataset.py:230`.
- EEG vectors are only filled if `eeg_masks[idx]` is true and the EEG row exists in `src/adagtcn_aligned/dataset.py:242`.
- `eeg_mask[idx] = 1.0` is set only when EEG is valid in `src/adagtcn_aligned/dataset.py:249`.
- Bridge summaries use `masked_mean(eeg, eeg_mask)` in `src/adagtcn_aligned/models.py:327`.

Caveat:

- The final sequence pooling uses `step_mask`, not modality masks. Thus real words with missing EEG/gaze still participate as zero-vector steps after fusion. This is not necessarily wrong, but missingness patterns can become a model cue. A modality-missingness sanity check is needed.

### 4.3 Gaze Missing Mask

**Verdict: mostly correct.**

Evidence:

- `gaze_masks` are read in `src/adagtcn_aligned/dataset.py:232`.
- Gaze vectors are filled only if `gaze_masks[idx]` is true in `src/adagtcn_aligned/dataset.py:236`.
- `gaze_mask[idx] = 1.0` is set in `src/adagtcn_aligned/dataset.py:240`.
- Bridge summaries use `masked_mean(gaze, gaze_mask)` in `src/adagtcn_aligned/models.py:328`.

### 4.4 `nFixations=0` Mask

**Verdict: correct at extraction time.**

Evidence:

- `has_positive_fixation()` checks FFD, GD, GPT, TRT, and nFixations for positive finite values in `src/adagtcn_aligned/extract_word_sequences.py:129`.
- After scalar gaze extraction, `gaze_valid` is overwritten by `has_positive_fixation(gaze)`, so non-positive or zero-fixation words are marked invalid.
- EEG is only extracted if `gaze_valid` is true, so skipped words do not receive EEG vectors in `src/adagtcn_aligned/extract_word_sequences.py:214` to `src/adagtcn_aligned/extract_word_sequences.py:218`.
- `gaze_masks.append(gaze_valid)` and `eeg_masks.append(has_eeg)` occur at `src/adagtcn_aligned/extract_word_sequences.py:222` and `src/adagtcn_aligned/extract_word_sequences.py:224`.

Full-data mask coverage:

| Split | Sequences | Words | Gaze word rate | EEG word rate | Zero-EEG sequence rate |
|---|---:|---:|---:|---:|---:|
| Train | 8,746 | 179,164 | 0.5997 | 0.5980 | 0.0067 |
| Validation | 1,260 | 25,758 | 0.5199 | 0.5190 | 0.0000 |
| Test | 1,478 | 30,276 | 0.6313 | 0.6313 | 0.0034 |

## 5. Real Differences Between Current Model Ablations

Model presets are defined in `src/adagtcn_aligned/models.py:340` to `src/adagtcn_aligned/models.py:348`. Training metrics are produced by `evaluate()` in `src/adagtcn_aligned/train_cnogsm.py:72`, using macro-F1 at `src/adagtcn_aligned/train_cnogsm.py:92`. Early stopping uses validation macro-F1 at `src/adagtcn_aligned/train_cnogsm.py:152`.

### Full-Data Results (`paper_y16_full`, seed 0)

| Model | Actual Difference | Test Acc | Test Macro-F1 | Test BAcc | Test AUROC | Evidence Strength |
|---|---|---:|---:|---:|---:|---|
| `adagtcn_aligned` | EEG graph + gaze encoder + TCN temporal encoder; no SSM, no bipartite, no bridge, no common-unique | 0.5277 | 0.3454 | 0.5000 | 0.5000 | Weak baseline; `loss=nan` |
| `eeg_graph_ssm` | EEG graph + SSM temporal encoder; gaze zeroed | 0.6252 | 0.5952 | 0.6422 | 0.7416 | Useful isolation evidence |
| `gaze_only_ssm` | Gaze encoder + SSM; EEG zeroed | 0.6495 | 0.6361 | 0.6622 | 0.6969 | Strong baseline evidence |
| `gaze_control_ssm` | EEG graph + gaze encoder + gaze-controlled SSM; no bipartite/bridge/common-unique | 0.6103 | 0.5764 | 0.6279 | 0.7496 | Mixed; worse than both single-modality SSMs in macro-F1 |
| `full_cnogsm` | EEG graph + gaze + SSM + bipartite + bridge + common-unique + auxiliary regularizers | 0.6658 | 0.6544 | 0.6779 | 0.7200 | Best full-data result, but modules are confounded |

### Pilot-Only Results (`pilot_y16_60sent`, seed 0 unless noted)

These are useful for debugging but should not be treated as final paper results.

| Model | Test Macro-F1 | Note |
|---|---:|---|
| `eeg_only_graph_tcn` | 0.4044 | Pilot only; EEG graph + TCN, gaze zeroed |
| `adagtcn_aligned` | 0.3333 | Pilot seed0; seed1 also 0.3333 |
| `gaze_control_ssm` | 0.6331 | Pilot only |
| `bipartite_graph_ssm` | 0.6119 | Pilot only; not run in full-data table |
| `bridge_bipartite_ssm` | 0.6215 | Pilot only; not run in full-data table |
| `full_cnogsm` | 0.7168 seed0 / 0.6267 seed1 | Pilot two-seed partial evidence |

### Important Baseline Problem

The current `adagtcn_aligned` baseline cannot be used as a strong proof of innovation because its full-data training history reports `loss=nan`. Its metrics match majority-class behavior: test accuracy equals the test majority accuracy (0.5277), balanced accuracy is 0.5000, and AUROC is 0.5000.

This means the current effect audit in `reports/adagtcn_aligned/paper_y16_full/cnogsm_ablation_effect_audit.md` is only an audit against a weak local baseline, not proof against AdaGTCN or a stable TCN implementation.

## 6. Is Each Module A Substantive Innovation Or A Classifier Change?

### `EEGGraphEncoder`

- Code: `src/adagtcn_aligned/models.py:77`.
- Role: learned adaptive graph over 105 EEG electrodes.
- Judgment: **not independently novel enough**, because AdaGTCN already uses graph modeling. It is a necessary aligned component, not a clean main contribution unless compared against a no-graph EEG-SSM.
- Current evidence: `eeg_graph_ssm` is strong, but no no-graph baseline exists.
- Required proof: `eeg_graph_ssm` vs `eeg_mlp_ssm` or `eeg_no_graph_ssm`, same split/seeds.
- If no improvement: do not claim EEG graph as contribution.

### `GazeControlledStateSpace`

- Code: `src/adagtcn_aligned/models.py:157`.
- Role: state-space temporal modeling over word/fixation order, with gaze vector entering the state update.
- Judgment: **most plausible substantive innovation**, but current proof is incomplete.
- Solves AdaGTCN limitation: replaces rigid TCN temporal convolution with recurrent state-space dynamics for reading sequences.
- Current evidence: `eeg_graph_ssm` full-data macro-F1 0.5952; `gaze_only_ssm` 0.6361; `gaze_control_ssm` 0.5764.
- Concern: `gaze_control_ssm` is worse than both `eeg_graph_ssm` and `gaze_only_ssm` in full-data macro-F1, so the "gaze-controlled fusion" part is not yet proven.
- Required proof:
  - Stable TCN baseline without NaN.
  - Vanilla SSM without gaze-control vs gaze-controlled SSM.
  - Multi-seed comparison.
- If no improvement over vanilla SSM: claim state-space temporal modeling, not gaze-control.

### `BipartiteNeuroOculomotorGraph`

- Code: `src/adagtcn_aligned/models.py:114`.
- Role: adds gaze node and gaze-to-EEG gating before graph convolution.
- Judgment: **substantive idea but currently evidence insufficient**.
- Solves AdaGTCN limitation: AdaGTCN mainly models EEG electrode graph; this attempts EEG-oculomotor interaction graph.
- Current evidence: only pilot evidence; no full-data result in `paper_y16_full`.
- Required proof:
  - Full-data `gaze_control_ssm` vs `bipartite_graph_ssm`.
  - Same seeds and metrics.
  - Optional graph interpretability: learned gaze-electrode interaction stability.
- If no improvement: delete from main contribution or move to appendix as negative/interpretability result.

### `SubjectInvariantBridge`

- Code: `src/adagtcn_aligned/models.py:211`.
- Loss: `subject_logits` adversarial term in `src/adagtcn_aligned/train_cnogsm.py:51` to `src/adagtcn_aligned/train_cnogsm.py:53`; reconstruction losses in `compute_aux_loss()` at `src/adagtcn_aligned/train_cnogsm.py:47`.
- Judgment: **not yet proven**.
- Solves AdaGTCN limitation: intended to reduce subject-specific nuisance information in subject-independent transfer.
- Current evidence: only pilot `bridge_bipartite_ssm`; full-data `full_cnogsm` includes bridge but does not isolate it.
- Required proof:
  - `bipartite_graph_ssm` vs `bridge_bipartite_ssm` full-data.
  - Subject classifier accuracy should decrease or subject leakage metric should improve.
  - Test macro-F1/BAcc should not drop.
- If no improvement: remove from main contribution.

### `CommonUniqueDisentangler`

- Code: `src/adagtcn_aligned/models.py:185`.
- Role: common/unique EEG-gaze latent decomposition with alignment/decorrelation losses.
- Judgment: **currently unproven**.
- Solves AdaGTCN limitation: attempts structured multimodal fusion rather than naive concatenation/gating.
- Current evidence: only included inside `full_cnogsm`; no `full_without_common_unique` result.
- Required proof:
  - `full_cnogsm` vs `full_no_common_unique`.
  - Report common alignment and unique decorrelation losses.
  - Modality-drop or modality-permutation sanity checks.
- If no improvement: delete from main contribution.

### `full_cnogsm`

- Code preset: `src/adagtcn_aligned/models.py:348`.
- Judgment: **promising system, not a clean single innovation yet**.
- Current evidence: best full-data result, macro-F1 0.6544.
- Problem: full model mixes SSM, bipartite graph, subject bridge, common-unique disentanglement, and regularizers, so its improvement cannot be assigned to any one module.
- Required proof:
  - Multi-seed full table.
  - Full model minus each module.
  - Stronger stable baseline.

## 7. Experiments Required To Prove Each Module

| Module | Required Ablation | Success Standard | Failure Interpretation |
|---|---|---|---|
| State-space temporal modeling | stable `eeg_only_graph_tcn` vs `eeg_graph_ssm`; stable `adagtcn_aligned` vs SSM variant | multi-seed macro-F1 and BAcc improve; no NaN loss | If not better than TCN, do not claim temporal innovation |
| Gaze-controlled update | vanilla SSM with EEG+gaze but no gaze-control vs `gaze_control_ssm` | gaze-control improves over vanilla SSM | If not, claim SSM only |
| EEG graph | `eeg_no_graph_ssm` vs `eeg_graph_ssm` | graph improves EEG-only performance | If not, graph is not a contribution |
| Eye-tracking sequence | `gaze_only_mlp`/aggregate gaze vs `gaze_only_ssm` | temporal gaze sequence improves over aggregate gaze | If not, eye-tracking is a strong baseline, not a sequence innovation |
| Bipartite graph | `gaze_control_ssm` vs `bipartite_graph_ssm` | full-data multi-seed improvement | If not, remove from main text |
| Subject bridge | `bipartite_graph_ssm` vs `bridge_bipartite_ssm` plus subject leakage metric | task performance improves and subject leakage decreases | If not, remove or relegate to appendix |
| Common-unique | `full_no_common_unique` vs `full_cnogsm` | full model improves and modality-drop robustness improves | If not, remove from contribution |
| Multimodal fusion | `eeg_graph_ssm`, `gaze_only_ssm`, and `full_cnogsm` | full exceeds both single-modality baselines | Current full-data result passes by small margin over gaze-only |

## 8. Modules With Insufficient Evidence For Main Contribution

Do not write these as main contributions yet:

1. `BipartiteNeuroOculomotorGraph`: no full-data result in `paper_y16_full`; pilot only.
2. `SubjectInvariantBridge`: no full-data isolated bridge ablation; no subject leakage metric.
3. `CommonUniqueDisentangler`: no no-common-unique ablation.
4. "AdaGTCN-aligned baseline improvement": current baseline has `loss=nan`, so improvement over it is not a publishable proof.
5. "Raw EEG waveform modeling": current input is word-level EEG band representations, not raw waveforms.

## 9. Most Likely Publishable Main Innovation Points

### Candidate 1: State-Space Temporal Modeling For Reading Sequences

Most likely to survive review if the TCN baseline is repaired. It directly addresses AdaGTCN-style temporal modeling and has strong empirical signal through `eeg_graph_ssm` and `gaze_only_ssm`.

Current best wording:

> We model ZuCo word-level EEG band representations and eye-tracking observations as reading sequences using state-space temporal encoders under an AdaGTCN-aligned available-Y subject-independent protocol.

### Candidate 2: Strong Oculomotor Sequence Baseline And Multimodal Complementarity

`gaze_only_ssm` is strong (macro-F1 0.6361), and `full_cnogsm` improves to 0.6544. This supports a cautious multimodal claim, but the improvement over gaze-only is modest and currently single-seed.

Current best wording:

> Eye-tracking sequence dynamics are a strong predictor of NR/TSR reading task; EEG graph state representations provide complementary information when combined in the full model.

### Candidate 3: Word-Level EEG Band Graph-State Representation

This is potentially useful but not independently proven. It needs a no-graph baseline before being elevated to a contribution.

## Final Audit Judgment

The current project is moving toward a verifiable contribution, but it is not yet ready to claim a full CNO-GSM module stack as publishable innovation. The safest research story is:

1. Use the correct protocol name: **AdaGTCN-aligned available-Y subject-independent protocol**.
2. Use the correct input name: **word-level EEG band representations and eye-tracking sequences**.
3. Treat `gaze_only_ssm` as a strong required baseline, not a side result.
4. Treat state-space temporal modeling as the leading innovation candidate.
5. Treat bipartite graph, subject bridge, and common-unique disentanglement as unproven until isolated full-data ablations are run.
6. Fix the `adagtcn_aligned` NaN-loss baseline before making any comparative claim against AdaGTCN-style models.

