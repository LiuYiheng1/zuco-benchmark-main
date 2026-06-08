import numpy as np
import os

def analyze_sentence_matching():
    subject = 'YAC'

    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'

    eeg_feats = np.load(eeg_path, allow_pickle=True).item()
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()

    eeg_by_sent = {}
    for key in eeg_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            sent_idx = int(parts[2])
            label = parts[1]
            if sent_idx not in eeg_by_sent:
                eeg_by_sent[sent_idx] = {'NR': [], 'TSR': [], 'keys': []}
            eeg_by_sent[sent_idx][label].append(key)
            eeg_by_sent[sent_idx]['keys'].append(key)

    gaze_by_sent = {}
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            sent_idx = int(parts[2])
            label = parts[1]
            if sent_idx not in gaze_by_sent:
                gaze_by_sent[sent_idx] = {'NR': [], 'TSR': [], 'keys': []}
            gaze_by_sent[sent_idx][label].append(key)
            gaze_by_sent[sent_idx]['keys'].append(key)

    common_sents = set(eeg_by_sent.keys()) & set(gaze_by_sent.keys())

    print(f"EEG sentences: {len(eeg_by_sent)}")
    print(f"Gaze sentences: {len(gaze_by_sent)}")
    print(f"Common sentences: {len(common_sents)}")

    print("\nLabel consistency check per sentence:")
    label_mismatch = 0
    label_match = 0
    for sent_idx in common_sents:
        eeg_labels = set()
        for key in eeg_by_sent[sent_idx]['keys']:
            if 'NR' in key:
                eeg_labels.add('NR')
            else:
                eeg_labels.add('TSR')

        gaze_labels = set()
        for key in gaze_by_sent[sent_idx]['keys']:
            if 'NR' in key:
                gaze_labels.add('NR')
            else:
                gaze_labels.add('TSR')

        if eeg_labels != gaze_labels:
            label_mismatch += 1
            print(f"  MISMATCH at sentence {sent_idx}: EEG={eeg_labels}, Gaze={gaze_labels}")
        else:
            label_match += 1

    print(f"\nLabel match: {label_match}, Label mismatch: {label_mismatch}")

    print("\n\nCorrect data extraction approach:")
    print("=" * 60)

    X_eeg = []
    X_gaze = []
    y = []
    aligned_keys = []

    for sent_idx in common_sents:
        eeg_labels = set()
        for key in eeg_by_sent[sent_idx]['keys']:
            if 'NR' in key:
                eeg_labels.add('NR')
            else:
                eeg_labels.add('TSR')

        gaze_labels = set()
        for key in gaze_by_sent[sent_idx]['keys']:
            if 'NR' in key:
                gaze_labels.add('NR')
            else:
                gaze_labels.add('TSR')

        if eeg_labels == gaze_labels:
            label = 0 if 'NR' in eeg_labels else 1

            eeg_data = np.array(eeg_feats[eeg_by_sent[sent_idx]['keys'][0]])
            if eeg_data[-1] in ['NR', 'TSR']:
                eeg_data = eeg_data[:-1]
            eeg_data = eeg_data.astype(float)

            gaze_data = np.array(gaze_feats[gaze_by_sent[sent_idx]['keys'][0]])
            if gaze_data[-1] in ['NR', 'TSR']:
                gaze_data = gaze_data[:-1]
            gaze_data = gaze_data.astype(float)

            X_eeg.append(eeg_data)
            X_gaze.append(gaze_data)
            y.append(label)
            aligned_keys.append((sent_idx, eeg_by_sent[sent_idx]['keys'][0], gaze_by_sent[sent_idx]['keys'][0]))

    X_eeg = np.array(X_eeg)
    X_gaze = np.array(X_gaze)
    y = np.array(y)

    print(f"\nCorrectly aligned:")
    print(f"  X_eeg shape: {X_eeg.shape}")
    print(f"  X_gaze shape: {X_gaze.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  NR count: {np.sum(y == 0)}, TSR count: {np.sum(y == 1)}")
    print(f"  NR %: {np.mean(y == 0) * 100:.2f}%")

    print("\nFirst 10 aligned keys:")
    for i, (sent_idx, eeg_key, gaze_key) in enumerate(aligned_keys[:10]):
        label = 'NR' if y[i] == 0 else 'TSR'
        print(f"  [{i}] sent={sent_idx}, label={label}, EEG={eeg_key}, Gaze={gaze_key}")

analyze_sentence_matching()