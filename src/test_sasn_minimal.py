"""Minimal SASN test"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

FEATURES_DIR = "features"
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

print('Loading data...')
X_train, y_train = load_eeg_data('YAC')
X_test, y_test = load_eeg_data('YAG')
print('YAC: X=' + str(X_train.shape) + ', y=' + str(len(y_train)))
print('YAG: X=' + str(X_test.shape) + ', y=' + str(len(y_test)))

print('\nSimple train/test split test...')
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
clf.fit(X_train_s, y_train)
preds = clf.predict(X_test_s)
acc = accuracy_score(y_test, preds)
print('YAG test accuracy: ' + str(round(acc, 4)))

print('\nTest passed!')