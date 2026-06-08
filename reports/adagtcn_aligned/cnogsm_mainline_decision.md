# CNO-GSM Mainline Module Decision

Scope: this decision is based on the current codebase and existing ablation design/results under the AdaGTCN-aligned available-Y subject-independent protocol. It does not claim reproduction of the original AdaGTCN 12/2/4 setting because the current project has 16 labeled Y subjects.

EEG inputs should be described as word-level EEG band representations, not raw EEG waveform sequences.

## Executive Decision

The CNO-GSM paper mainline should keep at most one module as a current main contribution:

1. **Gaze-Controlled State Space Module**: keep as the primary mainline contribution, but phrase the claim conservatively as state-space reading dynamics with gaze-conditioned transitions.

The following modules should not be claimed as main contributions in the current evidence state:

- **Neuro-Oculomotor Bipartite Dynamic Graph**: supplementary ablation candidate.
- **Cross-Subject Brain Bridge**: supplementary/auxiliary regularizer candidate.
- **Common-Unique Multimodal Contrastive Learning**: auxiliary loss only; current code is not enough to support a main contrastive-learning claim.
- **Causal Invariant Reading Factor**: future work/drop from current mainline.
- **EEGPT-style Masked Neuro Pretraining**: future work/drop from current mainline.

`full_cnogsm` is currently the best full-data tested model among the available reported runs, but its improvement is entangled across multiple modules. Therefore, it can be reported as the best system result, while the paper's main mechanistic contribution should not claim that every internal module is independently proven.

## Evidence Snapshot

Current full-data results in `reports/adagtcn_aligned/paper_y16_full/cnogsm_ablation_effect_audit.csv`:

| Model | Test macro-F1 | Role |
|---|---:|---|
| `adagtcn_aligned` | 0.3454 | AdaGTCN-aligned baseline; unstable/weak in this run |
| `eeg_graph_ssm` | 0.5952 | EEG-only state-space/graph candidate |
| `gaze_only_ssm` | 0.6361 | gaze-only state-space candidate |
| `gaze_control_ssm` | 0.5764 | compact multimodal gaze-controlled SSM |
| `full_cnogsm` | 0.6544 | best tested full system |

Pilot results show isolated `bipartite_graph_ssm` and `bridge_bipartite_ssm` ablations, but the full-data table does not yet include them. This prevents claiming them as main contributions.

## Mainline Module Table

| Candidate module | Code status | Current evidence | Decision | Paper role | Required ablation before stronger claim | Reason |
|---|---|---|---|---|---|---|
| Gaze-Controlled State Space Module | Implemented in `src/adagtcn_aligned/models.py`, `GazeControlledStateSpace`; used by presets such as `gaze_control_ssm`, `eeg_graph_ssm`, `gaze_only_ssm`, `full_cnogsm` | Full-data ablations exist, but `gaze_control_ssm` is weaker than `gaze_only_ssm` and `eeg_graph_ssm`; `full_cnogsm` is best | **Keep** | Primary mainline contribution, conservatively stated | TCN vs vanilla SSM vs gaze-conditioned SSM; EEG-only SSM vs gaze-only SSM vs multimodal gaze-controlled SSM; seed stability | Clear structural difference from AdaGTCN temporal convolution; current code supports it; no external pretraining; interpretable as gaze-conditioned reading dynamics |
| Neuro-Oculomotor Bipartite Dynamic Graph | Implemented in `src/adagtcn_aligned/models.py`, `BipartiteNeuroOculomotorGraph`; preset `bipartite_graph_ssm` | Pilot-only isolated evidence; missing full-data isolated ablation | **Supplementary** | Supplementary ablation, not main contribution yet | Full-data `gaze_control_ssm` vs `bipartite_graph_ssm`; edge dropout/random graph sanity check; parameter-matched control | It changes EEG-gaze interaction structure, but current evidence is insufficient for a main claim |
| Cross-Subject Brain Bridge | Implemented in `src/adagtcn_aligned/models.py`, `SubjectInvariantBridge`; preset `bridge_bipartite_ssm` | Pilot-only isolated evidence; no full-data isolated proof; possible over-regularization | **Supplementary / auxiliary regularizer** | Supplementary subject-invariance experiment | Full-data `bipartite_graph_ssm` vs `bridge_bipartite_ssm`; leave-one-subject diagnostics; subject-ID predictability before/after bridge | Relevant to subject-independent setting, but not yet proven to improve generalization |
| Common-Unique Multimodal Contrastive Learning | Implemented as `CommonUniqueDisentangler` in `src/adagtcn_aligned/models.py`; appears in `full_cnogsm` | No isolated full-data ablation; current implementation is disentanglement/alignment/decorrelation, not enough to claim full contrastive learning | **Auxiliary loss** | Auxiliary objective or supplementary ablation | `full_cnogsm` with/without common-unique losses; shuffled-pair sanity check; loss-weight sweep | The claim is too broad for the current evidence and name; keep only if it improves under controlled ablation |
| Causal Invariant Reading Factor | No clear implemented class/function corresponding to this module in current audited files | No ablation and no concrete causal intervention/invariance protocol | **Future work / drop** | Future work only | Need explicit environment definition, invariance penalty, intervention/swap sanity check, and subject/domain split proof | Cannot be a main contribution without code and causal verification protocol |
| EEGPT-style Masked Neuro Pretraining | No clear implemented pretraining pipeline in current audited files | No pretraining result; would likely require external or additional large-scale pretraining | **Future work / drop** | Future work only | Need pretraining task, pretrain corpus, no-leak split, downstream fine-tuning control | Violates the current criterion of not depending on external large-scale pretraining |

## Recommended Main Model

Because `full_cnogsm` is currently the best full-data tested model, it can remain the reported best system under the current results:

- Main system name: `full_cnogsm`
- Full-data test macro-F1: 0.6544
- Interpretation: best observed integrated system, not proof that every component is independently effective

However, the paper should not make all six modules into main contributions. The safest mainline is:

1. **Primary contribution**: state-space reading dynamics with gaze-conditioned transition.
2. **System result**: integrated CNO-GSM achieves the best current result under the AdaGTCN-aligned available-Y subject-independent protocol.
3. **Supplementary analysis**: bipartite graph, bridge, and common-unique objectives are tested as optional components.

If later full-data ablations show that `full_cnogsm` is not the best model, the paper should switch the main model to the best compact ablation rather than preserving the full name. Based on current full-data evidence, the compact fallback order is:

1. `gaze_only_ssm` if the goal is pure performance among compact tested models.
2. `eeg_graph_ssm` if the goal is EEG-centered comparison against AdaGTCN.
3. `gaze_control_ssm` only if later ablations show stable multimodal gains over both unimodal SSM controls.

## Module-Specific Decisions

### 1. Gaze-Controlled State Space Module

Decision: **Keep as main contribution.**

Code location:

- `src/adagtcn_aligned/models.py`, `GazeControlledStateSpace`
- `src/adagtcn_aligned/models.py`, model presets including `eeg_graph_ssm`, `gaze_only_ssm`, `gaze_control_ssm`, `full_cnogsm`

Difference from AdaGTCN:

- AdaGTCN-style modeling is centered on graph/temporal convolution.
- This module changes the temporal modeling assumption to recurrent/state-space reading dynamics.
- The gaze-conditioned transition gives eye-tracking features a mechanistic role in controlling temporal evolution rather than merely being concatenated as classifier input.

Required ablation:

- `adagtcn_aligned` vs `eeg_graph_ssm`
- `eeg_graph_ssm` vs `gaze_control_ssm`
- `gaze_only_ssm` vs `gaze_control_ssm`
- TCN control vs vanilla SSM vs gaze-conditioned SSM
- Multiple seeds and per-subject reporting

Success standard:

- `gaze_control_ssm` should outperform parameter-matched TCN/vanilla SSM controls.
- It should also beat or match the stronger unimodal control, otherwise the claim must be narrowed to state-space modeling rather than multimodal gaze control.

Failure mode:

- Current full-data result shows `gaze_control_ssm` below `gaze_only_ssm` and `eeg_graph_ssm`, so the gaze-control part is not independently proven yet.
- If this persists, keep the state-space contribution but remove any claim that gaze-controlled multimodal fusion is the source of improvement.

## 2. Neuro-Oculomotor Bipartite Dynamic Graph

Decision: **Supplementary ablation, not current main contribution.**

Code location:

- `src/adagtcn_aligned/models.py`, `BipartiteNeuroOculomotorGraph`
- `src/adagtcn_aligned/models.py`, preset `bipartite_graph_ssm`

Difference from AdaGTCN:

- This module models EEG-band and eye-tracking streams as two interacting node sets.
- This is more structurally meaningful than simple EEG-gaze concatenation.

Required ablation:

- Full-data `gaze_control_ssm` vs `bipartite_graph_ssm`
- Randomized bipartite edges vs learned bipartite edges
- Parameter-matched fusion control

Success standard:

- The bipartite model must improve full-data macro-F1 over a parameter-matched multimodal SSM and remain stable across seeds.

Failure mode:

- It may overfit because the available-Y protocol has only 16 labeled subjects.
- Without full-data isolated evidence, it is not safe to claim this as a main innovation.

## 3. Cross-Subject Brain Bridge

Decision: **Supplementary or auxiliary regularizer.**

Code location:

- `src/adagtcn_aligned/models.py`, `SubjectInvariantBridge`
- `src/adagtcn_aligned/models.py`, preset `bridge_bipartite_ssm`

Difference from AdaGTCN:

- It explicitly targets subject-independent alignment, which is relevant to the current protocol.

Required ablation:

- Full-data `bipartite_graph_ssm` vs `bridge_bipartite_ssm`
- Subject-ID predictability before/after bridge
- Per-held-out-subject performance variance

Success standard:

- It should improve held-out-subject macro-F1 and reduce between-subject performance variance without collapsing class information.

Failure mode:

- A bridge can remove discriminative subject-specific signal.
- It can also look helpful on aggregate while hurting minority subjects or unstable folds.

## 4. Common-Unique Multimodal Contrastive Learning

Decision: **Auxiliary loss only.**

Code location:

- `src/adagtcn_aligned/models.py`, `CommonUniqueDisentangler`
- Included in `full_cnogsm`

Difference from AdaGTCN:

- It attempts to separate shared and modality-specific representations.

Required ablation:

- `full_cnogsm` with common-unique loss disabled
- Common-only, unique-only, and decorrelation-only controls
- EEG-gaze shuffled-pair sanity check

Success standard:

- It must improve full-data macro-F1 or calibration without harming subject-independent generalization.

Failure mode:

- The current implementation does not by itself justify the phrase "contrastive learning" unless explicit positives/negatives and a contrastive objective are verified.
- If it only adds loss terms without improving controlled ablations, it should be removed from the main paper.

## 5. Causal Invariant Reading Factor

Decision: **Future work/drop from current mainline.**

Code status:

- No audited class/function currently supports a concrete causal invariant module.

Why it is not mainline:

- There is no explicit environment definition, intervention design, invariance objective, or causal sanity check.
- A causal claim would require substantially stronger protocol evidence than the current ablation design provides.

Required future evidence:

- Explicit subject/task/domain environments
- Invariance penalty or causal factorization
- Intervention or counterfactual sanity check
- Proof that gains are not from capacity or regularization alone

## 6. EEGPT-style Masked Neuro Pretraining

Decision: **Future work/drop from current mainline.**

Code status:

- No audited pretraining pipeline currently supports this claim.

Why it is not mainline:

- It would introduce dependence on additional pretraining data or a separate pretraining stage.
- The current user criterion excludes main contributions that depend on external large-scale pretraining.

Required future evidence:

- Clearly separated pretraining corpus
- No subject/data leakage into downstream test subjects
- Masking objective and reconstruction target
- Fine-tuning comparison against no-pretraining control

## Final Recommendation

Use this mainline:

| Role | Module/model |
|---|---|
| Main contribution | Gaze-Controlled State Space Module, conservatively framed as state-space reading dynamics with gaze-conditioned transitions |
| Best reported system | `full_cnogsm`, because it is currently the best full-data tested model |
| Supplementary ablation | Neuro-Oculomotor Bipartite Dynamic Graph |
| Supplementary/auxiliary | Cross-Subject Brain Bridge |
| Auxiliary loss only | Common-Unique disentanglement/alignment objective |
| Future work/drop | Causal Invariant Reading Factor |
| Future work/drop | EEGPT-style Masked Neuro Pretraining |

The paper should not preserve the full CNO-GSM acronym by force. If future full-data ablations show that a compact model outperforms `full_cnogsm`, the compact ablation should become the main model, and the unused modules should be moved to supplementary or removed.
