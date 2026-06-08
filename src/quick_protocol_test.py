"""Quick LOSO and Split experiments"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def get_trial_id(key):
    return f"{key.split('_')[0]}_{key.split('_')[1]}_{key.split('_')[2]}"

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
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
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

print("Loading data...", flush=True)
all_data = {}
for subj in Y_SUBJECTS:
    Xe, ye, tid_e = load_eeg_data(subj)
    Xg, yg, tid_g = load_gaze_features(subj)
    if Xe is not None and Xg is not None:
        Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
        all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
print(f"Loaded {len(all_data)} subjects", flush=True)

print("\n=== PROTOCOL 1: LOSO ===", flush=True)
results_loso = []
for held_out in Y_SUBJECTS:
    if held_out not in all_data:
        continue
    train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]
    X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
    y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
    X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

    X_eeg_test = all_data[held_out]['Xe']
    y_test = all_data[held_out]['ye']
    X_gaze_test = all_data[held_out]['Xg']

    row = {'subject': held_out}

    most_common = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
    row['Majority'] = accuracy_score(y_test, np.ones(len(y_test)) * most_common)

    scaler_e = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_e_s, y_train)
    probs = clf.predict_proba(X_e_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['EEG_SVM'] = accuracy_score(y_test, preds)
    row['EEG_SVM_f1'] = f1_score(y_test, preds, average='macro')
    row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
    row['EEG_SVM_auroc'] = roc_auc_score(y_test, probs)

    scaler_g = StandardScaler()
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_g_s, y_train)
    probs = clf.predict_proba(X_g_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['Gaze_SVM'] = accuracy_score(y_test, preds)
    row['Gaze_SVM_f1'] = f1_score(y_test, preds, average='macro')
    row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
    row['Gaze_SVM_auroc'] = roc_auc_score(y_test, probs)

    scaler_e = StandardScaler()
    scaler_g = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(np.hstack([X_e_s, X_g_s]), y_train)
    probs = clf.predict_proba(np.hstack([X_e_test_s, X_g_test_s]))[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['EEG+Gaze_concat'] = accuracy_score(y_test, preds)
    row['EEG+Gaze_concat_f1'] = f1_score(y_test, preds, average='macro')
    row['EEG+Gaze_concat_bacc'] = balanced_accuracy_score(y_test, preds)
    row['EEG+Gaze_concat_auroc'] = roc_auc_score(y_test, probs)

    scaler_e = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_e_s, y_train)
    probs = clf.predict_proba(X_e_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['EEG_MLP'] = accuracy_score(y_test, preds)
    row['EEG_MLP_f1'] = f1_score(y_test, preds, average='macro')
    row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
    row['EEG_MLP_auroc'] = roc_auc_score(y_test, probs)

    scaler_g = StandardScaler()
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    clf.fit(X_g_s, y_train)
    probs = clf.predict_proba(X_g_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['Gaze_MLP'] = accuracy_score(y_test, preds)
    row['Gaze_MLP_f1'] = f1_score(y_test, preds, average='macro')
    row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
    row['Gaze_MLP_auroc'] = roc_auc_score(y_test, probs)

    pca_models = {}
    for c in [0, 1]:
        X_c = X_eeg_train[y_train == c]
        if len(X_c) > 20:
            pca = PCA(n_components=20, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca

    def compute_errors(X, pms):
        err = np.zeros((len(X), len(pms) * 2))
        for i, (c, pca) in enumerate(pms.items()):
            if pca is not None:
                X_rec = pca.inverse_transform(pca.transform(X))
                e = X - X_rec
                err[:, i] = np.sqrt(np.sum(e ** 2, axis=1))
                err[:, 1 + i] = np.mean(np.abs(e), axis=1)
        return err

    err_train = compute_errors(X_eeg_train, pca_models)
    err_test = compute_errors(X_eeg_test, pca_models)
    scaler = StandardScaler()
    Xc = np.hstack([scaler.fit_transform(X_eeg_train), err_train])
    Xt = np.hstack([scaler.transform(X_eeg_test), err_test])
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(Xc, y_train)
    preds = clf.predict(Xt)
    probs = np.zeros((len(preds), 2))
    probs[preds == 0, 0] = 1.0
    probs[preds == 1, 1] = 1.0
    row['PCET_source'] = accuracy_score(y_test, preds)
    row['PCET_source_f1'] = f1_score(y_test, preds, average='macro')
    row['PCET_source_bacc'] = balanced_accuracy_score(y_test, preds)
    row['PCET_source_auroc'] = roc_auc_score(y_test, probs[:, 1])

    scaler_e = StandardScaler()
    scaler_g = StandardScaler()
    X_e_s = scaler_e.fit_transform(X_eeg_train)
    X_e_test_s = scaler_e.transform(X_eeg_test)
    X_g_s = scaler_g.fit_transform(X_gaze_train)
    X_g_test_s = scaler_g.transform(X_gaze_test)

    gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    gaze_mlp.fit(X_g_s, y_train)
    z_gaze = gaze_mlp.predict_proba(X_g_s)
    entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
    confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
    attention = entropy * 0.01 + confidence
    att_tiled = np.tile(attention, (1, X_e_s.shape[1]))
    X_e_att = X_e_s * att_tiled

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_e_att, y_train)

    z_gaze_test = gaze_mlp.predict_proba(X_g_test_s)
    entropy_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)
    confidence_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)
    attention_test = entropy_test * 0.01 + confidence_test
    att_tiled_test = np.tile(attention_test, (1, X_e_test_s.shape[1]))
    X_e_att_test = X_e_test_s * att_tiled_test

    probs = clf.predict_proba(X_e_att_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['GETA_source'] = accuracy_score(y_test, preds)
    row['GETA_source_f1'] = f1_score(y_test, preds, average='macro')
    row['GETA_source_bacc'] = balanced_accuracy_score(y_test, preds)
    row['GETA_source_auroc'] = roc_auc_score(y_test, probs)

    eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    eeg_mlp.fit(X_e_s, y_train)
    z_eeg = eeg_mlp.predict_proba(X_e_s)
    z_eeg_test = eeg_mlp.predict_proba(X_e_test_s)

    alpha = 1 / (1 + np.exp(-z_eeg[:, 0] + z_gaze[:, 0]))
    z_fused = alpha.reshape(-1, 1) * z_eeg + (1 - alpha.reshape(-1, 1)) * z_gaze

    alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))
    z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

    clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
    clf_final.fit(z_fused, y_train)
    probs = clf_final.predict_proba(z_fused_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row['PCET+GETA+CAGF'] = accuracy_score(y_test, preds)
    row['PCET+GETA+CAGF_f1'] = f1_score(y_test, preds, average='macro')
    row['PCET+GETA+CAGF_bacc'] = balanced_accuracy_score(y_test, preds)
    row['PCET+GETA+CAGF_auroc'] = roc_auc_score(y_test, probs)

    results_loso.append(row)
    print(f"  {held_out}: CAGF={row['PCET+GETA+CAGF']*100:.1f}%", flush=True)

df_loso = pd.DataFrame(results_loso)
df_loso.to_csv(os.path.join(RESULTS_DIR, 'benchmark_style_loso_results.csv'), index=False)

print("\n=== LOSO SUMMARY ===", flush=True)
for col in ['Majority', 'EEG_SVM', 'Gaze_SVM', 'EEG+Gaze_concat', 'EEG_MLP', 'Gaze_MLP', 'PCET_source', 'GETA_source', 'PCET+GETA+CAGF']:
    if col in df_loso.columns:
        print(f"  {col}: {df_loso[col].mean()*100:.1f}±{df_loso[col].std()*100:.1f}%")

print("\n=== PROTOCOL 2: 12/2/4 SPLIT ===", flush=True)
np.random.seed(0)
shuffled = Y_SUBJECTS.copy()
np.random.shuffle(shuffled)
train = shuffled[:12]
val = shuffled[12:14]
test = shuffled[14:]
print(f"Train: {train}", flush=True)
print(f"Val: {val}", flush=True)
print(f"Test: {test}", flush=True)

X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train])
y_train = np.concatenate([all_data[s]['ye'] for s in train])
X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train])
X_eeg_test = np.vstack([all_data[s]['Xe'] for s in test])
y_test = np.concatenate([all_data[s]['ye'] for s in test])
X_gaze_test = np.vstack([all_data[s]['Xg'] for s in test])

row = {'split': '12train/2val/4test'}

most_common = 1 if np.sum(y_train == 1) >= np.sum(y_train == 0) else 0
row['Majority'] = accuracy_score(y_test, np.ones(len(y_test)) * most_common)

scaler_e = StandardScaler()
X_e_s = scaler_e.fit_transform(X_eeg_train)
X_e_test_s = scaler_e.transform(X_eeg_test)
clf = SVC(kernel='rbf', probability=True, random_state=42)
clf.fit(X_e_s, y_train)
probs = clf.predict_proba(X_e_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)
row['EEG_SVM'] = accuracy_score(y_test, preds)
row['EEG_SVM_f1'] = f1_score(y_test, preds, average='macro')
row['EEG_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
row['EEG_SVM_auroc'] = roc_auc_score(y_test, probs)

scaler_g = StandardScaler()
X_g_s = scaler_g.fit_transform(X_gaze_train)
X_g_test_s = scaler_g.transform(X_gaze_test)
clf = SVC(kernel='rbf', probability=True, random_state=42)
clf.fit(X_g_s, y_train)
probs = clf.predict_proba(X_g_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)
row['Gaze_SVM'] = accuracy_score(y_test, preds)
row['Gaze_SVM_f1'] = f1_score(y_test, preds, average='macro')
row['Gaze_SVM_bacc'] = balanced_accuracy_score(y_test, preds)
row['Gaze_SVM_auroc'] = roc_auc_score(y_test, probs)

scaler_e = StandardScaler()
scaler_g = StandardScaler()
X_e_s = scaler_e.fit_transform(X_eeg_train)
X_e_test_s = scaler_e.transform(X_eeg_test)
X_g_s = scaler_g.fit_transform(X_gaze_train)
X_g_test_s = scaler_g.transform(X_gaze_test)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(np.hstack([X_e_s, X_g_s]), y_train)
probs = clf.predict_proba(np.hstack([X_e_test_s, X_g_test_s]))[:, 1]
preds = (probs >= 0.5).astype(int)
row['EEG+Gaze_concat'] = accuracy_score(y_test, preds)
row['EEG+Gaze_concat_f1'] = f1_score(y_test, preds, average='macro')
row['EEG+Gaze_concat_bacc'] = balanced_accuracy_score(y_test, preds)
row['EEG+Gaze_concat_auroc'] = roc_auc_score(y_test, probs)

scaler_e = StandardScaler()
X_e_s = scaler_e.fit_transform(X_eeg_train)
X_e_test_s = scaler_e.transform(X_eeg_test)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X_e_s, y_train)
probs = clf.predict_proba(X_e_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)
row['EEG_MLP'] = accuracy_score(y_test, preds)
row['EEG_MLP_f1'] = f1_score(y_test, preds, average='macro')
row['EEG_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
row['EEG_MLP_auroc'] = roc_auc_score(y_test, probs)

scaler_g = StandardScaler()
X_g_s = scaler_g.fit_transform(X_gaze_train)
X_g_test_s = scaler_g.transform(X_gaze_test)
clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
clf.fit(X_g_s, y_train)
probs = clf.predict_proba(X_g_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)
row['Gaze_MLP'] = accuracy_score(y_test, preds)
row['Gaze_MLP_f1'] = f1_score(y_test, preds, average='macro')
row['Gaze_MLP_bacc'] = balanced_accuracy_score(y_test, preds)
row['Gaze_MLP_auroc'] = roc_auc_score(y_test, probs)

pca_models = {}
for c in [0, 1]:
    X_c = X_eeg_train[y_train == c]
    if len(X_c) > 20:
        pca = PCA(n_components=20, random_state=42)
        pca.fit(X_c)
        pca_models[c] = pca

err_train = compute_errors(X_eeg_train, pca_models)
err_test = compute_errors(X_eeg_test, pca_models)
scaler = StandardScaler()
Xc = np.hstack([scaler.fit_transform(X_eeg_train), err_train])
Xt = np.hstack([scaler.transform(X_eeg_test), err_test])
clf = RidgeClassifier(alpha=0.1)
clf.fit(Xc, y_train)
preds = clf.predict(Xt)
probs = np.zeros((len(preds), 2))
probs[preds == 0, 0] = 1.0
probs[preds == 1, 1] = 1.0
row['PCET_source'] = accuracy_score(y_test, preds)
row['PCET_source_f1'] = f1_score(y_test, preds, average='macro')
row['PCET_source_bacc'] = balanced_accuracy_score(y_test, preds)
row['PCET_source_auroc'] = roc_auc_score(y_test, probs[:, 1])

scaler_e = StandardScaler()
scaler_g = StandardScaler()
X_e_s = scaler_e.fit_transform(X_eeg_train)
X_e_test_s = scaler_e.transform(X_eeg_test)
X_g_s = scaler_g.fit_transform(X_gaze_train)
X_g_test_s = scaler_g.transform(X_gaze_test)

gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
gaze_mlp.fit(X_g_s, y_train)
z_gaze = gaze_mlp.predict_proba(X_g_s)
z_gaze_test = gaze_mlp.predict_proba(X_g_test_s)
entropy = -np.sum(z_gaze * np.log(z_gaze + 1e-8), axis=1).reshape(-1, 1)
confidence = np.max(z_gaze, axis=1).reshape(-1, 1)
attention = entropy * 0.01 + confidence
att_tiled = np.tile(attention, (1, X_e_s.shape[1]))
X_e_att = X_e_s * att_tiled
entropy_test = -np.sum(z_gaze_test * np.log(z_gaze_test + 1e-8), axis=1).reshape(-1, 1)
confidence_test = np.max(z_gaze_test, axis=1).reshape(-1, 1)
attention_test = entropy_test * 0.01 + confidence_test
att_tiled_test = np.tile(attention_test, (1, X_e_test_s.shape[1]))
X_e_att_test = X_e_test_s * att_tiled_test

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X_e_att, y_train)
probs = clf.predict_proba(X_e_att_test)[:, 1]
preds = (probs >= 0.5).astype(int)
row['GETA_source'] = accuracy_score(y_test, preds)
row['GETA_source_f1'] = f1_score(y_test, preds, average='macro')
row['GETA_source_bacc'] = balanced_accuracy_score(y_test, preds)
row['GETA_source_auroc'] = roc_auc_score(y_test, probs)

eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
eeg_mlp.fit(X_e_s, y_train)
z_eeg = eeg_mlp.predict_proba(X_e_s)
z_eeg_test = eeg_mlp.predict_proba(X_e_test_s)

alpha = 1 / (1 + np.exp(-z_eeg[:, 0] + z_gaze[:, 0]))
z_fused = alpha.reshape(-1, 1) * z_eeg + (1 - alpha.reshape(-1, 1)) * z_gaze
alpha_test = 1 / (1 + np.exp(-z_eeg_test[:, 0] + z_gaze_test[:, 0]))
z_fused_test = alpha_test.reshape(-1, 1) * z_eeg_test + (1 - alpha_test.reshape(-1, 1)) * z_gaze_test

clf_final = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
clf_final.fit(z_fused, y_train)
probs = clf_final.predict_proba(z_fused_test)[:, 1]
preds = (probs >= 0.5).astype(int)
row['PCET+GETA+CAGF'] = accuracy_score(y_test, preds)
row['PCET+GETA+CAGF_f1'] = f1_score(y_test, preds, average='macro')
row['PCET+GETA+CAGF_bacc'] = balanced_accuracy_score(y_test, preds)
row['PCET+GETA+CAGF_auroc'] = roc_auc_score(y_test, probs)

df_split = pd.DataFrame([row])
df_split.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_style_split_results.csv'), index=False)

print("\n=== SPLIT SUMMARY ===", flush=True)
for col in ['Majority', 'EEG_SVM', 'Gaze_SVM', 'EEG+Gaze_concat', 'EEG_MLP', 'Gaze_MLP', 'PCET_source', 'GETA_source', 'PCET+GETA+CAGF']:
    if col in df_split.columns:
        val = df_split[col].values[0]
        print(f"  {col}: {val*100:.1f}%")

print("\nDone!", flush=True)