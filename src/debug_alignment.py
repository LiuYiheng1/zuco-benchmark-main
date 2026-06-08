import os
import numpy as np
FEATURES_DIR = "features"
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def get_trial_id(key):
    parts = key.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def load_gaze_features(subject):
    GAZE_FEATURE_FILE = 'sent_gaze_sacc'
    path = os.path.join(FEATURES_DIR, f"{subject}_{GAZE_FEATURE_FILE}.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        numeric_vals = [float(v) for v in values[:-1]]
        features = np.array(numeric_vals, dtype=np.float64)
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def align_eeg_gaze(X_eeg, y_eeg, trial_ids_eeg, X_gaze, y_gaze, trial_ids_gaze):
    gaze_dict = {tid: (X_gaze[i], y_gaze[i]) for i, tid in enumerate(trial_ids_gaze)}
    X_eeg_aligned, y_eeg_aligned, X_gaze_aligned, y_gaze_aligned = [], [], [], []
    for i, tid in enumerate(trial_ids_eeg):
        if tid in gaze_dict:
            X_eeg_aligned.append(X_eeg[i])
            y_eeg_aligned.append(y_eeg[i])
            X_gaze_aligned.append(gaze_dict[tid][0])
            y_gaze_aligned.append(gaze_dict[tid][1])
    return (np.array(X_eeg_aligned), np.array(y_eeg_aligned),
            np.array(X_gaze_aligned), np.array(y_gaze_aligned))

# Test alignment for YAC
Xe, ye, tid_e = load_eeg_data('YAC')
Xg, yg, tid_g = load_gaze_features('YAC')
print(f'EEG: {Xe.shape}, Gaze: {Xg.shape}')
Xe_a, ye_a, Xg_a, yg_a = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
print(f'Aligned EEG: {Xe_a.shape}, Aligned Gaze: {Xg_a.shape}')

# Test on all subjects
for subj in Y_SUBJECTS[:3]:
    Xe, ye, tid_e = load_eeg_data(subj)
    Xg, yg, tid_g = load_gaze_features(subj)
    if Xe is None or Xg is None:
        print(f'{subj}: data not found')
        continue
    Xe_a, ye_a, Xg_a, yg_a = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
    print(f'{subj}: EEG {Xe.shape}->{Xe_a.shape}, Gaze {Xg.shape}->{Xg_a.shape}')