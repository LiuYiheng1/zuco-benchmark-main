import numpy as np
import os

def debug_data_extraction():
    SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG']

    for subject in SUBJECTS:
        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print("="*60)

        eeg_path = f'features/{subject}_electrode_features_all.npy'
        gaze_path = f'features/{subject}_sent_gaze_sacc.npy'

        if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
            print("Files not found!")
            continue

        eeg_feats = np.load(eeg_path, allow_pickle=True).item()
        gaze_feats = np.load(gaze_path, allow_pickle=True).item()

        print(f"\nOriginal EEG dict: {len(eeg_feats)} entries")
        print(f"Original Gaze dict: {len(gaze_feats)} entries")

        nr_count_raw = 0
        tsr_count_raw = 0
        for key in eeg_feats.keys():
            if 'NR' in key:
                nr_count_raw += 1
            elif 'TSR' in key:
                tsr_count_raw += 1

        print(f"\nRaw counts from keys: NR={nr_count_raw}, TSR={tsr_count_raw}")

        nr_count_clean = 0
        tsr_count_clean = 0
        nr_last_val = []
        tsr_last_val = []

        for key in eeg_feats.keys():
            data = np.array(eeg_feats[key])
            last_val = data[-1]
            if 'NR' in key:
                nr_count_clean += 1
                nr_last_val.append(last_val)
            elif 'TSR' in key:
                tsr_count_clean += 1
                tsr_last_val.append(last_val)

        print(f"\nAfter removing last element: NR={nr_count_clean}, TSR={tsr_count_clean}")

        print(f"\nNR last values (first 5): {nr_last_val[:5]}")
        print(f"TSR last values (first 5): {tsr_last_val[:5]}")

        print(f"\nNR unique last values: {np.unique(nr_last_val)}")
        print(f"TSR unique last values: {np.unique(tsr_last_val)}")

        eeg_keys = set()
        for key in eeg_feats.keys():
            parts = key.split('_')
            if len(parts) >= 3:
                sentence_idx = int(parts[2])
                eeg_keys.add(sentence_idx)

        gaze_keys = set()
        for key in gaze_feats.keys():
            parts = key.split('_')
            if len(parts) >= 3:
                sentence_idx = int(parts[2])
                gaze_keys.add(sentence_idx)

        intersection = eeg_keys & gaze_keys

        print(f"\nEEG unique sentence indices: {len(eeg_keys)}")
        print(f"Gaze unique sentence indices: {len(gaze_keys)}")
        print(f"Intersection: {len(intersection)}")

        X_eeg = []
        y = []
        for idx in intersection:
            for key in eeg_feats.keys():
                parts = key.split('_')
                if len(parts) >= 3 and int(parts[2]) == idx:
                    data = np.array(eeg_feats[key])
                    if data[-1] in ['NR', 'TSR']:
                        data = data[:-1]
                    data = data.astype(float)
                    X_eeg.append(data)
                    y.append(0 if 'NR' in parts[1] else 1)
                    break

        X_eeg = np.array(X_eeg)
        y = np.array(y)

        print(f"\nExtracted EEG shape: {X_eeg.shape}")
        print(f"Extracted labels shape: {y.shape}")
        print(f"NR count: {np.sum(y == 0)}, TSR count: {np.sum(y == 1)}")
        print(f"NR %: {np.mean(y == 0) * 100:.2f}%")

debug_data_extraction()