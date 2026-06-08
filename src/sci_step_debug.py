"""SCI Step-by-Step Debug"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("Step-by-Step Debug")

FEATURES_DIR = "features"
Y_SUBJECTS = ['YAC', 'YAG', 'YAK']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
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

print("Step 1: Loading data...")
all_data = {}
for subj in Y_SUBJECTS:
    X, y = load_eeg_data(subj)
    all_data[subj] = (X, y)
    print(f"  {subj}: {X.shape}")

print("Step 2: Setting up experiment...")
held_out = 'YAC'
X_test_orig, y_test_orig = all_data[held_out]
train_subjs = [s for s in Y_SUBJECTS if s != held_out]

X_train_all = np.vstack([all_data[s][0] for s in train_subjs])
y_train_all = np.concatenate([all_data[s][1] for s in train_subjs])
print(f"  Train: {X_train_all.shape}, Test: {X_test_orig.shape}")

print("Step 3: Splitting data...")
n_samples = len(y_test_orig)
np.random.seed(0)
indices = np.random.permutation(n_samples)
test_size = n_samples // 3
test_indices = indices[:test_size]
cal_pool_indices = indices[test_size:]

X_test = X_test_orig[test_indices]
y_test = y_test_orig[test_indices]
X_cal_pool = X_test_orig[cal_pool_indices]
y_cal_pool = y_test_orig[cal_pool_indices]
print(f"  Test: {X_test.shape}, Cal pool: {X_cal_pool.shape}")

print("Step 4: Calibration sampling...")
n_cal = 3
class_0_idx = np.where(y_cal_pool == 0)[0]
class_1_idx = np.where(y_cal_pool == 1)[0]
np.random.shuffle(class_0_idx)
np.random.shuffle(class_1_idx)
n0 = min(n_cal, len(class_0_idx))
n1 = min(n_cal, len(class_1_idx))
selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
X_cal = X_cal_pool[selected]
y_cal = y_cal_pool[selected]
print(f"  Calibration: {X_cal.shape}")

print("Step 5: Computing features...")
mu_0 = np.mean(X_cal[y_cal == 0], axis=0)
mu_1 = np.mean(X_cal[y_cal == 1], axis=0)
sigma_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8
sigma_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8

scaler = StandardScaler()
X_cal_s = scaler.fit_transform(X_cal)
X_test_s = scaler.transform(X_test)
print(f"  Scaled: Cal={X_cal_s.shape}, Test={X_test_s.shape}")

print("Step 6: Computing Mahalanobis...")
sigma_0_inv = np.linalg.inv(np.diag(sigma_0 ** 2))
sigma_1_inv = np.linalg.inv(np.diag(sigma_1 ** 2))

d0_test = np.sqrt(np.sum((X_test_s - mu_0) * (np.dot(X_test_s - mu_0, sigma_0_inv)), axis=1))
d1_test = np.sqrt(np.sum((X_test_s - mu_1) * (np.dot(X_test_s - mu_1, sigma_1_inv)), axis=1))
uncertainty_test = np.abs(d1_test - d0_test)
print(f"  Uncertainty: {uncertainty_test.shape}")

print("Step 7: Training classifiers...")
clf = LogisticRegression(max_iter=200, solver='lbfgs')
clf.fit(X_cal_s, y_cal)
p_pcet = clf.predict_proba(X_test_s)[:, 1]
print(f"  PCET done")

clf_unc = LogisticRegression(max_iter=200, solver='lbfgs')
clf_unc.fit(np.column_stack([uncertainty_test[:len(y_cal)]]), y_cal)
p_srgc = clf_unc.predict_proba(np.column_stack([uncertainty_test]))[:, 1]
print(f"  SRGC done")

print("Step 8: Evaluating...")
acc_pcet = accuracy_score(y_test, (p_pcet >= 0.5).astype(int))
acc_srgc = accuracy_score(y_test, (p_srgc >= 0.5).astype(int))
print(f"  PCET: {acc_pcet:.4f}, SRGC: {acc_srgc:.4f}")

print("Done!")