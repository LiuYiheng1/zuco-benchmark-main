import numpy as np
import os

def investigate_gaze_structure():
    subject = 'YAC'

    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()

    print("Investigating Gaze data structure")
    print("=" * 60)

    sent_0_data = []
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3 and int(parts[2]) == 0:
            sent_0_data.append(key)

    print(f"\nSentence 0 has {len(sent_0_data)} entries:")
    for key in sent_0_data:
        data = np.array(gaze_feats[key])
        print(f"  {key}: shape={data.shape}, last_val={data[-1]}")

    sent_1_data = []
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3 and int(parts[2]) == 1:
            sent_1_data.append(key)

    print(f"\nSentence 1 has {len(sent_1_data)} entries:")
    for key in sent_1_data:
        data = np.array(gaze_feats[key])
        print(f"  {key}: shape={data.shape}, last_val={data[-1]}")

    print("\n\nInvestigating the 4th number in keys:")
    print("Format: SUBJ_LABEL_SENT_TRIAL")
    print("For EEG: trial=0,1,2,3...")
    print("For Gaze: trial=0,1,2,3... (but gaze has both NR/TSR for same sentence)")

    gaze_by_sent_label = {}
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 4:
            sent_idx = int(parts[2])
            trial_idx = int(parts[3])
            label = parts[1]
            key_str = f"{sent_idx}_{label}"
            if key_str not in gaze_by_sent_label:
                gaze_by_sent_label[key_str] = []
            gaze_by_sent_label[key_str].append((key, trial_idx))

    print(f"\nGaze has {len(gaze_by_sent_label)} unique (sent, label) combinations")

    print("\n\nHypothesis: The gaze file contains ALL sentences with BOTH labels")
    print("This might be because gaze was recorded independently of EEG")

    double_label_sents = 0
    for key, entries in gaze_by_sent_label.items():
        labels = set()
        for entry in entries:
            for k in gaze_feats.keys():
                if k == entry[0]:
                    labels.add(k.split('_')[1])
        if len(labels) > 1 or len(entries) > 1:
            double_label_sents += 1

    print(f"\nSentences with multiple entries: {double_label_sents}")

    print("\n\nConclusion:")
    print("The gaze file seems to be a COMPLETE list of sentences with both labels")
    print("For each sentence, there are entries for BOTH NR and TSR (regardless of actual condition)")
    print("This is likely gaze data aggregated across all sentences, not condition-specific")

    print("\n\nCorrect approach:")
    print("1. Use EEG keys to determine the ACTUAL label for each sentence")
    print("2. Match gaze data by sentence index ONLY")
    print("3. If gaze has both NR and TSR for a sentence, use the one matching EEG's label")

    print("\n\nRe-doing alignment with correct approach:")
    eeg_path = f'features/{subject}_electrode_features_all.npy'
    eeg_feats = np.load(eeg_path, allow_pickle=True).item()

    eeg_by_sent = {}
    for key in eeg_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            sent_idx = int(parts[2])
            label = 0 if 'NR' in parts[1] else 1
            if sent_idx not in eeg_by_sent:
                eeg_by_sent[sent_idx] = {'label': label, 'key': key}

    gaze_by_sent_label = {}
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            sent_idx = int(parts[2])
            label_str = parts[1]
            if sent_idx not in gaze_by_sent_label:
                gaze_by_sent_label[sent_idx] = {}
            gaze_by_sent_label[sent_idx][label_str] = key

    X_eeg = []
    X_gaze = []
    y = []

    for sent_idx, eeg_info in eeg_by_sent.items():
        if sent_idx in gaze_by_sent_label:
            label = eeg_info['label']
            label_str = 'NR' if label == 0 else 'TSR'

            if label_str in gaze_by_sent_label[sent_idx]:
                gaze_key = gaze_by_sent_label[sent_idx][label_str]

                eeg_data = np.array(eeg_feats[eeg_info['key']])
                if eeg_data[-1] in ['NR', 'TSR']:
                    eeg_data = eeg_data[:-1]
                eeg_data = eeg_data.astype(float)

                gaze_data = np.array(gaze_feats[gaze_key])
                if gaze_data[-1] in ['NR', 'TSR']:
                    gaze_data = gaze_data[:-1]
                gaze_data = gaze_data.astype(float)

                X_eeg.append(eeg_data)
                X_gaze.append(gaze_data)
                y.append(label)

    X_eeg = np.array(X_eeg)
    X_gaze = np.array(X_gaze)
    y = np.array(y)

    print(f"\nCorrectly aligned:")
    print(f"  X_eeg shape: {X_eeg.shape}")
    print(f"  X_gaze shape: {X_gaze.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  NR count: {np.sum(y == 0)}, TSR count: {np.sum(y == 1)}")
    print(f"  NR %: {np.mean(y == 0) * 100:.2f}%")

investigate_gaze_structure()