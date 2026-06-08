import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def get_trial_id(key):
    return f"{key.split('_')[0]}_{key.split('_')[1]}_{key.split('_')[2]}"

def load_eeg_data(subject):
    path = os.path.join('features', f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split('_')
        if len(parts) >= 2 and parts[1] == 'NR':
            label = 1
        elif len(parts) >= 2 and parts[1] == 'TSR':
            label = 0
        else:
            continue
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def load_gaze_features(subject):
    path = os.path.join('features', f"{subject}_sent_gaze_sacc.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split('_')
        if len(parts) >= 2 and parts[1] == 'NR':
            label = 1
        elif len(parts) >= 2 and parts[1] == 'TSR':
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

print("Loading data...")
all_data = {}
for subj in Y_SUBJECTS:
    Xe, ye, tid_e = load_eeg_data(subj)
    Xg, yg, tid_g = load_gaze_features(subj)
    if Xe is not None and Xg is not None:
        Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
        all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
        print(f'{subj}: EEG {Xe_a.shape}, Gaze {Xg_a.shape}')
    else:
        print(f'{subj}: FAILED')

print(f'\nTotal subjects loaded: {len(all_data)}')

if len(all_data) >= 2:
    subjs = list(all_data.keys())
    train_subjs = subjs[:15]
    test_subj = subjs[15]

    X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
    y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
    X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

    X_eeg_test = all_data[test_subj]['Xe']
    y_test = all_data[test_subj]['ye']
    X_gaze_test = all_data[test_subj]['Xg']

    print(f'\nTrain: {len(y_train)} samples, Test: {len(y_test)} samples')

    print('\nTesting SVM...')
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_eeg_train)
    X_test_s = scaler.transform(X_eeg_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print(f'EEG_SVM Acc: {accuracy_score(y_test, preds):.3f}')

    print('\nTesting MLP...')
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print(f'EEG_MLP Acc: {accuracy_score(y_test, preds):.3f}')

    print('\nTesting PCET...')
    for c in [0, 1]:
        X_c = X_eeg_train[y_train == c]
        if len(X_c) > 20:
            pca = PCA(n_components=20, random_state=42)
            pca.fit(X_c)
            X_rec = pca.inverse_transform(pca.transform(X_eeg_test))
            e = X_eeg_test - X_rec
            print(f'PCET class {c} error shape: {e.shape}')

    print('\nAll tests passed!')
