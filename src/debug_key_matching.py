import numpy as np
import os

def debug_key_matching():
    subject = 'YAC'

    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'

    eeg_feats = np.load(eeg_path, allow_pickle=True).item()
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()

    print("Sample EEG keys:")
    for i, key in enumerate(list(eeg_feats.keys())[:10]):
        print(f"  {key}")

    print("\nSample Gaze keys:")
    for i, key in enumerate(list(gaze_feats.keys())[:10]):
        print(f"  {key}")

    print("\n\nParsing EEG keys:")
    eeg_parsed = []
    for key in eeg_feats.keys():
        parts = key.split('_')
        if len(parts) >= 4:
            subj = parts[0]
            label = parts[1]
            sent_idx = int(parts[2])
            trial_idx = int(parts[3])
            eeg_parsed.append((key, subj, label, sent_idx, trial_idx))

    print(f"\nEEG trials per sentence (first 5 sentences):")
    sent_trials = {}
    for key, subj, label, sent_idx, trial_idx in eeg_parsed:
        if sent_idx not in sent_trials:
            sent_trials[sent_idx] = []
        sent_trials[sent_idx].append((key, label, trial_idx))

    for sent_idx in sorted(list(sent_trials.keys()))[:5]:
        trials = sent_trials[sent_idx]
        nr_trials = [t for t in trials if t[1] == 'NR']
        tsr_trials = [t for t in trials if t[1] == 'TSR']
        print(f"  Sentence {sent_idx}: {len(trials)} trials ({len(nr_trials)} NR, {len(tsr_trials)} TSR)")

    print("\n\nParsing Gaze keys:")
    gaze_parsed = []
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 4:
            subj = parts[0]
            label = parts[1]
            sent_idx = int(parts[2])
            trial_idx = int(parts[3])
            gaze_parsed.append((key, subj, label, sent_idx, trial_idx))

    print(f"\nGaze trials per sentence (first 5 sentences):")
    gaze_sent_trials = {}
    for key, subj, label, sent_idx, trial_idx in gaze_parsed:
        if sent_idx not in gaze_sent_trials:
            gaze_sent_trials[sent_idx] = []
        gaze_sent_trials[sent_idx].append((key, label, trial_idx))

    for sent_idx in sorted(list(gaze_sent_trials.keys()))[:5]:
        trials = gaze_sent_trials[sent_idx]
        nr_trials = [t for t in trials if t[1] == 'NR']
        tsr_trials = [t for t in trials if t[1] == 'TSR']
        print(f"  Sentence {sent_idx}: {len(trials)} trials ({len(nr_trials)} NR, {len(tsr_trials)} TSR)")

    print("\n\nFull key matching check:")
    eeg_keys_set = set(eeg_feats.keys())
    gaze_keys_set = set(gaze_feats.keys())
    full_intersection = eeg_keys_set & gaze_keys_set

    print(f"EEG keys: {len(eeg_keys_set)}")
    print(f"Gaze keys: {len(gaze_keys_set)}")
    print(f"Full key intersection: {len(full_intersection)}")

    print("\nSample intersecting keys:")
    for key in list(full_intersection)[:10]:
        print(f"  {key}")

debug_key_matching()