"""SCI Debug - Minimal Test"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("Debug Test")

FEATURES_DIR = "features"
Y_SUBJECTS = ['YAC', 'YAG', 'YAK']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
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

print("Loading YAC...")
X, y = load_eeg_data('YAC')
if X is not None:
    print(f"  Loaded: X={X.shape}, y={y.shape}")
    print(f"  Classes: {np.unique(y)}")

    print("\nRunning simple classification...")
    np.random.seed(0)
    indices = np.random.permutation(len(y))
    train_idx = indices[:10]
    test_idx = indices[10:30]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)

    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"  Accuracy: {acc:.4f}")
    print("  Done!")
else:
    print("  Failed to load!")