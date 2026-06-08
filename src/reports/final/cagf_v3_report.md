# CAGF-v3 Cross-Interaction Fusion Report

## Results

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| EEG+Gaze_concat | 57.7¡À7.9 | 61.5¡À7.3 | 66.1¡À7.2 | 72.0¡À7.0 | 79.4¡À6.1 || Static_average | 46.5¡À14.0 | 49.3¡À15.8 | 64.3¡À15.1 | 65.7¡À16.5 | 79.7¡À7.0 || CAGF_feature_only | 62.3¡À9.2 | 65.8¡À9.5 | 68.9¡À9.5 | 72.9¡À8.6 | 78.6¡À7.6 || CAGF_without_confidence | 62.3¡À9.2 | 65.8¡À9.5 | 68.9¡À9.5 | 72.9¡À8.6 | 78.6¡À7.6 || CAGF_full_old | 60.9¡À7.7 | 63.7¡À8.4 | 67.7¡À8.8 | 72.2¡À7.5 | 78.6¡À6.5 || CAGF_v3_cross_interaction | 61.6¡À9.1 | 64.2¡À9.3 | 68.5¡À9.0 | 72.7¡À8.1 | 77.1¡À7.3 |
## Success Criteria

| Criterion | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot | Result |
|-----------|--------|--------|---------|---------|---------|--------|
| v3 >= feature_only | FAIL | FAIL | FAIL | FAIL | FAIL | PASS || v3 > concat | PASS | PASS | PASS | PASS | FAIL | PASS || v3 > static | PASS | PASS | PASS | PASS | FAIL | PASS |
## Conclusion

CAGF-v3 (Cross-modal Adaptive Gated Fusion) uses cross-modal interaction features:
- abs_diff = |z_eeg - z_gaze|: disagreement magnitude
- hadamard = z_eeg * z_gaze: co-activation pattern

Gate input: concat([z_eeg, z_gaze, abs_diff, hadamard])
Alpha = sigmoid(MLP(gate_input))
z_fused = alpha * z_eeg + (1-alpha) * z_gaze

Passes all criteria in 0/5 shots.
