import os, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime

FEATURES_DIR = 'features'
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
RESULTS_DIR = 'results/loso'
os.makedirs(RESULTS_DIR, exist_ok=True)

log_file = open(os.path.join(RESULTS_DIR, 'loso_log.txt'), 'w')
log_file.write('Starting LOSO-Y SVM EEG\n')
log_file.flush()

def log(msg):
    log_file.write(msg + '\n')
    log_file.flush()
    print(msg)

def load_features(subject, feature_name):
    path = os.path.join(FEATURES_DIR, subject + '_' + feature_name + '.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split('_')
    if len(parts) >= 2 and parts[1] == 'NR':
        return 'NR', True
    elif len(parts) >= 2 and parts[1] == 'TSR':
        return 'TSR', True
    return '', False

def load_labeled_data(subjects, feature_name):
    all_X, all_y = [], []
    for subj in subjects:
        feats = load_features(subj, feature_name)
        if feats is None:
            continue
        for key, values in feats.items():
            label, is_labeled = parse_key(key)
            if not is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            label_binary = 1 if label == 'NR' else 0
            all_X.append(features)
            all_y.append(label_binary)
    return np.array(all_X), np.array(all_y)

log('LOSO-Y SVM EEG (16 subjects, 1 seed)')
results = []
for i, held_out in enumerate(Y_SUBJECTS):
    train_subjs = [s for s in Y_SUBJECTS if s != held_out]
    X_train, y_train = load_labeled_data(train_subjs, 'electrode_features_all')
    X_test, y_test = load_labeled_data([held_out], 'electrode_features_all')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(random_state=0, kernel='linear', gamma='scale')
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)

    results.append({'held_out': held_out, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'n_test': len(y_test)})
    log(str(i+1) + '/16 ' + held_out + ': Acc=' + str(round(acc,4)) + ', F1=' + str(round(f1,4)) + ', BAcc=' + str(round(bacc,4)))

results_df = pd.DataFrame(results)
acc_mean = results_df['accuracy'].mean()
acc_std = results_df['accuracy'].std()
log('Mean: Acc=' + str(round(acc_mean,4)) + '+-' + str(round(acc_std,4)))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fname = 'svm_eeg_loso_all_' + timestamp + '.csv'
results_df.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
log('Saved to ' + os.path.join(RESULTS_DIR, fname))
log_file.close()
print('Done!')