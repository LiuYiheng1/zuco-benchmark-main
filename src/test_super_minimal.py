"""Super Minimal Test"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Starting super minimal test...", flush=True)

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK']

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

results = []

for seed in [0]:
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

        np.random.seed(seed)
        indices = np.random.permutation(len(y_test_orig))
        test_size = len(y_test_orig) // 3
        test_indices = indices[:test_size]
        cal_pool_indices = indices[test_size:]

        X_test = X_test_orig[test_indices]
        y_test = y_test_orig[test_indices]
        X_cal_pool = X_test_orig[cal_pool_indices]
        y_cal_pool = y_test_orig[cal_pool_indices]

        n_cal = 5
        class_0_idx = np.where(y_cal_pool == 0)[0]
        class_1_idx = np.where(y_cal_pool == 1)[0]
        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)
        n0 = min(n_cal, len(class_0_idx))
        n1 = min(n_cal, len(class_1_idx))
        selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])

        X_cal = X_cal_pool[selected]
        y_cal = y_cal_pool[selected]

        scaler = StandardScaler()
        X_cal_s = scaler.fit_transform(X_cal)
        X_test_s = scaler.transform(X_test)

        clf = SVC(kernel='rbf', random_state=42)
        clf.fit(X_cal_s, y_cal)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            'seed': seed, 'subject': held_out, 'n_cal': n_cal,
            'method': 'RBF_SVM',
            'accuracy': acc
        })
        print(f"Subject {held_out}: acc={acc:.4f}", flush=True)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/super_minimal_test.csv", index=False)
print(f"Results saved! Total rows: {len(df)}", flush=True)
print("Done!", flush=True)