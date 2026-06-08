import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')

log_path = 'd:/pycharmproject/zuco-benchmark-main/src/results/final/sci_log2.txt'
log_file = open(log_path, 'w')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

log("Starting...")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

log("Imports done")

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
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
    return np.array(X), np.array(y)

log("Functions defined")

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

log("Starting main loop...")

for seed in seeds:
    log(f"\nSeed {seed}:")
    for held_out in Y_SUBJECTS:
        X_test_orig, y_test_orig = load_eeg_data(held_out)
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        print(f" {held_out}", end='', flush=True)

        for n_cal in shot_settings:
            cal_idx = np.random.choice(len(y_test_orig), n_cal*2, replace=False)
            X_cal = X_test_orig[cal_idx]
            y_cal = y_test_orig[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            scaler = StandardScaler()
            X_cal_s = scaler.fit_transform(X_cal)
            X_test_s = scaler.transform(X_test_orig)

            clf = LogisticRegression(max_iter=500, random_state=seed)
            clf.fit(X_cal_s, y_cal)
            p_test = clf.predict_proba(X_test_s)[:, 1]
            y_pred = (p_test >= 0.5).astype(int)

            acc = accuracy_score(y_test_orig, y_pred)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'SVM', 'accuracy': acc})

        print('.', end='', flush=True)

    df_partial = pd.DataFrame(results)
    df_partial.to_csv(f"{RESULTS_DIR}/sci_v2_partial.csv", index=False)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/sci_v2_results.csv", index=False)

log("\nDone!")
log_file.close()