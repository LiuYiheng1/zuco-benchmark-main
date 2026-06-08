# AdaGTCN Alignment Protocol

## Decision

- Use `Y16_12_2_2_seed0` as the primary available-Y subject split.
- Use `Y16_LOSO_*` folds as the stability protocol.
- Do not call the available-Y split `12/2/4`: only 16 labeled Y subjects are present.

## Why

The AdaGTCN paper-style 12 train / 2 validation / 4 test protocol needs 18 labeled subjects. This workspace has 16 labeled Y subjects with both NR and TSR raw Matlab files, so an exact 18-subject split cannot be created without additional labeled subjects or the original AdaGTCN subject list.

## Input Alignment

Protocol alignment alone is not enough. AdaGTCN-style comparison must use word/fixation-level EEG and eye-tracking sequences from the raw `.mat` files rather than only sentence-level precomputed feature vectors.
