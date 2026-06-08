# AdaGTCN-Aligned Innovation Modules

This note proposes modular, large-step extensions over AdaGTCN for ZuCo reading
task identification with EEG and eye-tracking data.

## Baseline To Beat

AdaGTCN uses word-level fixation-locked EEG sequences, adaptive graph learning,
graph convolution, and temporal convolution for NR vs TSR reading task
identification.

Local alignment rule: do not compare sentence-level proxy results as a full
AdaGTCN reproduction. Use the `src/adagtcn_aligned` word/fixation-level track.

## Proposed Main Model: Causal Neuro-Oculomotor Graph State Model

Short name: `CNO-GSM`.

Core claim: reading intent is not just an EEG spatial graph problem. It is a
controlled neural-oculomotor dynamical system where gaze exposes the control
policy, EEG exposes latent cognitive state, and subject identity is a nuisance
domain. The model should learn a subject-invariant latent state while preserving
modality-specific evidence.

## Module 1: Gaze-Controlled Neural State Space Encoder

AdaGTCN temporal modeling can be replaced or augmented with a selective state
space block over fixation/word order. The state update is conditioned on
eye-tracking variables such as fixation duration, regression, refixation count,
and pupil size.

Input:

- EEG node features per word/fixation.
- Eye-tracking control vector per word/fixation.
- Masks for skipped words and missing EEG.

Mechanism:

- Learn latent state `z_t`.
- Use gaze as a control input `u_t`.
- Update `z_t = SSM(z_{t-1}, EEG_t, u_t)`.
- Use selective gates so difficult or task-relevant fixations receive longer
  memory.

Why this is a big step:

- AdaGTCN treats temporal modeling as sequence convolution. This treats reading
  as a controlled dynamical system.
- The module is inspired by control theory and modern selective state-space
  sequence models.

## Module 2: Neuro-Oculomotor Bipartite Dynamic Graph

AdaGTCN learns EEG electrode adjacency. We extend the graph to two interacting
node sets:

- EEG electrode nodes.
- Oculomotor-event nodes: current word, fixation event, saccade/regression
  event, pupil/load event.

Edges:

- EEG-EEG: structural priors plus learned functional connectivity.
- Gaze-gaze: word order, fixation order, regression jumps.
- EEG-gaze: gaze-conditioned neural recruitment.

Why this is a big step:

- Eye movement is not only a side feature. It becomes a graph controller and a
  physiological evidence source.
- This directly targets eye-brain alignment.

## Module 3: Subject-Invariant Brain Bridge

Use a cross-subject bridge inspired by recent fMRI/brain-decoding work:

- Subject-specific lightweight adapters map each subject into a shared latent
  space.
- A shared encoder/classifier operates in that space.
- A cyclic reconstruction loss maps the shared latent back to subject-specific
  EEG/gaze statistics.

Losses:

- Task classification loss.
- Cross-subject contrastive loss for same task / similar text difficulty.
- Subject reconstruction cycle loss.
- Subject adversarial or domain confusion loss.

Why this is a big step:

- AdaGTCN has domain-independent components, but does not explicitly learn a
  reversible subject bridge.
- This can support both zero-shot and few-shot new-subject adaptation.

## Module 4: Causal Invariance And Domain Disentanglement

Separate three factors:

- `c`: causal reading-intent factor.
- `s`: subject-specific physiology.
- `x`: text/item difficulty and lexical confounds.

Training:

- Treat subject as environment.
- Treat sentence/item as a possible confound.
- Learn a representation whose classifier is stable across subjects.
- Penalize subject leakage while keeping modality-specific useful residuals.

Why this is a big step:

- It makes cross-subject generalization a causal/domain-generalization problem,
  not just a neural architecture problem.
- It helps avoid inflated gains from text or subject shortcuts.

## Module 5: Multimodal Common-Unique Contrastive Learning

Do not collapse EEG and eye-tracking into one early-fused vector. Learn:

- Common latent: shared cognitive-load / reading-intent evidence.
- EEG-unique latent: neural oscillatory and topographic evidence.
- Gaze-unique latent: oculomotor strategy evidence.

Losses:

- EEG-gaze common contrastive alignment.
- Modality-unique decorrelation.
- Task-supervised prototype contrast.
- Modality dropout and imagination for skipped words or missing signals.

Why this is a big step:

- Multimodal contrastive theory suggests the gain comes from cooperation between
  modalities, but naive alignment can suppress modality-specific signal.
- ZuCo has natural missingness/skipped words, so robust multimodal learning is
  not optional.

## Module 6: Physics/Neuroscience Priors For The EEG Graph

Use priors that are stronger than arbitrary learned adjacency:

- Spatial distance prior over electrodes.
- Hemispheric asymmetry features.
- Frequency-band-specific connectivity.
- Smoothness or Laplacian energy regularization on graph signals.
- Dynamic connectivity change penalty to avoid noisy adjacency jumps.

Why this is a big step:

- AdaGTCN learns adaptive adjacency but can overfit small subject counts.
- Priors can reduce variance and make learned graphs interpretable.

## Module 7: Text As A Controlled Confound, Not A Main Signal

Text should be handled carefully:

- Use text difficulty features only as controls or nuisance variables.
- Do not let text embeddings dominate classification.
- Use residualization: predict EEG/gaze representations after accounting for
  sentence length, word frequency, and readability.
- Report no-text and controlled-text variants.

Why this is a big step:

- NR and TSR may differ in materials and reading behavior. A strong paper needs
  to prove that gains come from physiological reading state, not text shortcuts.

## Recommended Ablation Path

1. AdaGTCN-aligned input extractor and Y16 split.
2. Reproduce an AdaGTCN-like graph-temporal baseline on word/fixation sequences.
3. Add Module 1 only: gaze-controlled state space temporal encoder.
4. Add Module 2: bipartite EEG-gaze graph.
5. Add Module 3 and 4: subject bridge plus causal invariance.
6. Add Module 5: common-unique contrastive pretraining.
7. Add Module 6 and 7 for final robustness and interpretability.

## Naming Candidates

- CNO-GSM: Causal Neuro-Oculomotor Graph State Model.
- BridgeRead: Cross-Subject Neuro-Oculomotor Bridge for Reading Intent.
- GazeCtrl-GSM: Gaze-Controlled Graph State Model.

Recommended name: `CNO-GSM`, because it captures causality, EEG/eye movement,
graph structure, and dynamics in one compact phrase.

