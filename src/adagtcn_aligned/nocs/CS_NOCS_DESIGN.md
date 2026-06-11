# CS-NOCS: Causally Safe Neuro-Oculomotor Control

## 1. Problem Diagnosis

The current word-level ZuCo results show a clear asymmetric modality problem.

- The exact gaze anchor is strong and reproducible:
  - AUROC = 0.6939 +/- 0.1660
  - Macro-F1 = 0.5739 +/- 0.1691
- EEG-only and naive EEG-gaze fusion are weak:
  - naive concat AUROC = 0.5369
  - EEG behaves as a high-noise neural observation rather than a stable predictive modality.
- NOCS full reduces naive fusion damage but does not reliably improve over the gaze anchor:
  - AUROC = 0.6922
- SafeNOCS validation admission is conservative but not fully safe:
  - AUROC = 0.6926
  - safe > anchor: 7 subjects
  - safe = anchor: 6 subjects
  - safe < anchor: 3 subjects

The core issue is not how to force EEG into the predictor. The core issue is how to admit EEG only when there is validation evidence that it improves the gaze anchor without causing subject-level negative transfer.

## 2. Theoretical Sources

CS-NOCS is a post-hoc admission layer inspired by four lines of work.

- Selective prediction and risk control:
  - Selective classifiers trade coverage for reliability and only make augmented predictions when a risk certificate is acceptable.
  - Conformal risk control motivates using validation/calibration evidence and conservative lower confidence bounds before trusting a higher-risk decision path.
- Gated multimodal fusion:
  - Gated multimodal units show that modality influence should be conditional, not unconditional.
  - CS-NOCS uses a hard admission gate at the protocol level and an error-aware gate at the sample level.
- Cross-subject functional alignment:
  - Neural signals are subject-specific and may require subject-aware reliability checks before cross-subject transfer.
  - CS-NOCS treats each LOSO protocol as a reliability test for whether EEG correction transfers to the held-out subject.
- Gaze-controlled state dynamics:
  - Gaze is the stable behavioral observation.
  - EEG is treated as a noisy correction signal that may perturb the gaze anchor only after passing a validation certificate.

Reference starting points:

- Selective Classification for Deep Neural Networks, NeurIPS 2017.
- SelectiveNet, ICML 2019.
- Conformal Risk Control, ICLR 2024.
- Selective Conformal Risk Control, 2025.
- Gated Multimodal Units for Information Fusion, 2017.
- Hyperalignment and cross-subject functional alignment in neuroscience.

## 3. Final Design

CS-NOCS has four pieces.

### Exact Gaze Anchor

The anchor is the already validated sklearn-style gaze LR:

- mask-aware gaze mean pooling
- mask-aware gaze std pooling
- valid_ratio
- sequence_length
- valid_count
- StandardScaler
- LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")

This anchor is the default prediction path.

### EEG Utility Certificate

For each LOSO protocol, validation predictions are used to estimate whether EEG correction is useful:

- anchor_val_auroc
- corrected_val_auroc
- delta_val = corrected_val_auroc - anchor_val_auroc
- bootstrap 95% CI for delta_val
- macro-F1 delta
- balanced-accuracy delta

EEG is admitted only if:

- delta_val_mean > 0
- bootstrap_CI_low >= -epsilon
- corrected_val_macro_f1 >= anchor_val_macro_f1 - 0.005
- corrected_val_balanced_acc >= anchor_val_balanced_acc - 0.005

If the certificate fails, the output is exactly the gaze anchor.

### Error-Aware EEG Correction

EEG should not directly replace gaze. It should only act where the anchor is uncertain and EEG disagrees enough to be informative.

For each sample:

- conf_anchor = max(p_anchor, 1 - p_anchor)
- disagreement = abs(p_corrected - p_anchor)
- risk = disagreement * (1 - conf_anchor)

Validation searches a threshold over risk quantiles and a small alpha grid. EEG correction applies only when risk exceeds the selected threshold.

### Conservative Fallback

All strategies include an anchor fallback. If validation evidence does not improve the anchor, prediction is:

```text
p_final = p_anchor
```

If EEG is admitted:

```text
p_final = (1 - alpha) * p_anchor + alpha * p_corrected
```

## 4. Explicit Safety Commitment

If EEG utility is not certified on the validation split, prediction must fall back to the exact gaze anchor. EEG is never allowed to participate by default.

The research objective is not to force CS-NOCS to exceed the gaze anchor. The objective is to remove or reduce EEG-induced negative transfer while preserving EEG's subject-dependent gains when validation evidence supports admission.
