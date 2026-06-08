import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

SUBJECTS_16 = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS',
                'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_aligned_eeg_gaze(subject):
    """
    Unified data loading function with correct EEG-gaze alignment.

    Key insight: Gaze file has BOTH NR and TSR entries for each sentence:
      YAC_NR_0_0 and YAC_TSR_0_250 both represent sentence 0 with different conditions.

    Correct approach:
      1. Use EEG keys as anchor (each EEG key = one unique trial with true label)
      2. Parse subject, label, sentence_id from EEG key
      3. Find gaze entry with SAME label + sentence_id
      4. Only keep samples where both exist and labels match

    Returns:
      X_eeg: EEG features array
      X_gaze: Gaze features array
      y: labels array (0=NR, 1=TSR)
      aligned_keys: list of (eeg_key, gaze_key) tuples
    """
    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'

    if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
        return None, None, None, None, {}

    eeg_feats = np.load(eeg_path, allow_pickle=True).item()
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()

    eeg_keys = list(eeg_feats.keys())

    gaze_by_label_sent = {}
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            label = parts[1]
            sent_idx = int(parts[2])
            gaze_by_label_sent[(label, sent_idx)] = key

    X_eeg = []
    X_gaze = []
    y = []
    aligned_keys = []

    label_match_count = 0
    label_mismatch_count = 0

    for eeg_key in eeg_keys:
        parts = eeg_key.split('_')
        if len(parts) < 3:
            continue

        subj = parts[0]
        label = parts[1]
        sent_idx = int(parts[2])

        gaze_key = gaze_by_label_sent.get((label, sent_idx))

        if gaze_key is None:
            continue

        eeg_data = np.array(eeg_feats[eeg_key])
        gaze_data = np.array(gaze_feats[gaze_key])

        if eeg_data[-1] in ['NR', 'TSR']:
            eeg_data = eeg_data[:-1]
        if gaze_data[-1] in ['NR', 'TSR']:
            gaze_data = gaze_data[:-1]

        eeg_data = eeg_data.astype(float)
        gaze_data = gaze_data.astype(float)

        if len(eeg_data) == 0 or len(gaze_data) == 0:
            continue
        if len(eeg_data.shape) != 1 or len(gaze_data.shape) != 1:
            continue

        gaze_label = gaze_key.split('_')[1]
        if label == gaze_label:
            label_match_count += 1
        else:
            label_mismatch_count += 1

        X_eeg.append(eeg_data)
        X_gaze.append(gaze_data)
        y.append(0 if label == 'NR' else 1)
        aligned_keys.append((eeg_key, gaze_key))

    if len(X_eeg) == 0:
        return None, None, None, None, {}

    X_eeg = np.array(X_eeg)
    X_gaze = np.array(X_gaze)
    y = np.array(y)

    total = label_match_count + label_mismatch_count
    label_consistency = label_match_count / total if total > 0 else 0

    info = {
        'subject': subject,
        'eeg_keys_count': len(eeg_keys),
        'gaze_keys_count': len(gaze_feats),
        'aligned_count': len(X_eeg),
        'label_match': label_match_count,
        'label_mismatch': label_mismatch_count,
        'label_consistency': label_consistency,
        'nr_count': int(np.sum(y == 0)),
        'tsr_count': int(np.sum(y == 1)),
        'nr_pct': float(np.mean(y == 0) * 100),
        'eeg_dim': X_eeg.shape[1] if len(X_eeg.shape) > 1 else 0,
        'gaze_dim': X_gaze.shape[1] if len(X_gaze.shape) > 1 else 0
    }

    return X_eeg, X_gaze, y, aligned_keys, info


def run_single_experiment(subject, X_eeg, X_gaze, y, k, seed, method):
    """
    Run a single experiment for one method.
    """
    np.random.seed(seed)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_idx, test_idx = next(sss.split(X_eeg, y))

    X_eeg_cal, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
    X_gaze_cal, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
    y_cal, y_test = y[train_idx], y[test_idx]

    cal_idx = []
    for c in [0, 1]:
        c_idx = np.where(y_cal == c)[0]
        if len(c_idx) >= k:
            selected = np.random.choice(c_idx, k, replace=False)
        else:
            selected = c_idx
        cal_idx.extend(selected)

    X_eeg_cal = X_eeg_cal[cal_idx]
    X_gaze_cal = X_gaze_cal[cal_idx]
    y_cal = y_cal[cal_idx]

    n_cal = len(y_cal)
    n_test = len(y_test)

    if method == 'EEG_SVM':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_eeg_cal)
        X_test = scaler.transform(X_eeg_test)
        clf = SVC(kernel='rbf', C=1.0)
        clf.fit(X_train, y_cal)
        y_pred = clf.predict(X_test)

    elif method == 'EEG_MLP':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_eeg_cal)
        X_test = scaler.transform(X_eeg_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
        clf.fit(X_train, y_cal)
        y_pred = clf.predict(X_test)

    elif method == 'Gaze_SVM':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_gaze_cal)
        X_test = scaler.transform(X_gaze_test)
        clf = SVC(kernel='rbf', C=1.0)
        clf.fit(X_train, y_cal)
        y_pred = clf.predict(X_test)

    elif method == 'Gaze_MLP':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_gaze_cal)
        X_test = scaler.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
        clf.fit(X_train, y_cal)
        y_pred = clf.predict(X_test)

    elif method == 'Raw_Fusion':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)
        X_concat_train = np.hstack([X_eeg_train, X_gaze_train])
        X_concat_test = np.hstack([X_eeg_test_s, X_gaze_test_s])
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
        clf.fit(X_concat_train, y_cal)
        y_pred = clf.predict(X_concat_test)

    elif method == 'Ridge_StaticAvg':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        ridge_eeg = RidgeClassifier(alpha=1.0)
        ridge_eeg.fit(X_eeg_train, y_cal)
        z_eeg_train = ridge_eeg.decision_function(X_eeg_train)
        z_eeg_test = ridge_eeg.decision_function(X_eeg_test_s)

        ridge_gaze = RidgeClassifier(alpha=1.0)
        ridge_gaze.fit(X_gaze_train, y_cal)
        z_gaze_train = ridge_gaze.decision_function(X_gaze_train)
        z_gaze_test = ridge_gaze.decision_function(X_gaze_test_s)

        z_avg_train = 0.5 * z_eeg_train + 0.5 * z_gaze_train
        z_avg_test = 0.5 * z_eeg_test + 0.5 * z_gaze_test

        clf = RidgeClassifier(alpha=0.1)
        clf.fit(z_avg_train.reshape(-1, 1), y_cal)
        y_pred = clf.predict(z_avg_test.reshape(-1, 1))

    elif method == 'PCET_only':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_eeg_cal)
        X_test_scaled = scaler.transform(X_eeg_test)

        pca_models = {}
        for c in [0, 1]:
            X_class = X_scaled[y_cal == c]
            if len(X_class) > 5:
                pca = PCA(n_components=min(5, len(X_class)-1))
                pca.fit(X_class)
                pca_models[c] = pca

        X_pcet = X_scaled
        X_pcet_test = X_test_scaled
        for c in pca_models:
            pca = pca_models[c]
            X_recon = pca.inverse_transform(pca.transform(X_scaled))
            X_pcet = np.hstack([X_pcet, np.abs(X_scaled - X_recon)])
            X_recon_t = pca.inverse_transform(pca.transform(X_test_scaled))
            X_pcet_test = np.hstack([X_pcet_test, np.abs(X_test_scaled - X_recon_t)])

        clf = RidgeClassifier(alpha=0.1)
        clf.fit(X_pcet, y_cal)
        y_pred = clf.predict(X_pcet_test)

    elif method == 'GETA_only':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        pca_models = {}
        for c in [0, 1]:
            X_class = X_eeg_train[y_cal == c]
            if len(X_class) > 5:
                pca = PCA(n_components=min(5, len(X_class)-1))
                pca.fit(X_class)
                pca_models[c] = pca

        X_eeg_enhanced = X_eeg_train
        X_eeg_test_enhanced = X_eeg_test_s
        for c in pca_models:
            pca = pca_models[c]
            X_recon = pca.inverse_transform(pca.transform(X_eeg_train))
            X_eeg_enhanced = np.hstack([X_eeg_enhanced, np.abs(X_eeg_train - X_recon)])
            X_recon_t = pca.inverse_transform(pca.transform(X_eeg_test_s))
            X_eeg_test_enhanced = np.hstack([X_eeg_test_enhanced, np.abs(X_eeg_test_s - X_recon_t)])

        gaze_mlp_train = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
        gaze_mlp_train.fit(X_gaze_train, y_cal)
        z_gaze_train = gaze_mlp_train.predict_proba(X_gaze_train)[:, 1]
        z_gaze_test = gaze_mlp_train.predict_proba(X_gaze_test_s)[:, 1]

        alpha_gaze = 1 / (1 + np.exp(-(z_gaze_train - 0.5)))
        alpha_gaze_test = 1 / (1 + np.exp(-(z_gaze_test - 0.5)))

        X_weighted = X_eeg_enhanced * (alpha_gaze.reshape(-1, 1) + 0.5)
        X_weighted_test = X_eeg_test_enhanced * (alpha_gaze_test.reshape(-1, 1) + 0.5)

        clf = RidgeClassifier(alpha=0.1)
        clf.fit(X_weighted, y_cal)
        y_pred = clf.predict(X_weighted_test)

    elif method == 'PCET+GETA+CAGF':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        pca_models = {}
        for c in [0, 1]:
            X_class = X_eeg_train[y_cal == c]
            if len(X_class) > 5:
                pca = PCA(n_components=min(5, len(X_class)-1))
                pca.fit(X_class)
                pca_models[c] = pca

        X_eeg_enhanced = X_eeg_train
        X_eeg_test_enhanced = X_eeg_test_s
        for c in pca_models:
            pca = pca_models[c]
            X_recon = pca.inverse_transform(pca.transform(X_eeg_train))
            X_eeg_enhanced = np.hstack([X_eeg_enhanced, np.abs(X_eeg_train - X_recon)])
            X_recon_t = pca.inverse_transform(pca.transform(X_eeg_test_s))
            X_eeg_test_enhanced = np.hstack([X_eeg_test_enhanced, np.abs(X_eeg_test_s - X_recon_t)])

        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
        gaze_mlp.fit(X_gaze_train, y_cal)
        z_gaze_train = gaze_mlp.predict_proba(X_gaze_train)[:, 1]
        z_gaze_test = gaze_mlp.predict_proba(X_gaze_test_s)[:, 1]

        alpha_gaze = 1 / (1 + np.exp(-(z_gaze_train - 0.5)))
        alpha_gaze_test = 1 / (1 + np.exp(-(z_gaze_test - 0.5)))

        X_weighted = X_eeg_enhanced * (alpha_gaze.reshape(-1, 1) + 0.5)
        X_weighted_test = X_eeg_test_enhanced * (alpha_gaze_test.reshape(-1, 1) + 0.5)

        ridge_pcet = RidgeClassifier(alpha=0.1)
        ridge_pcet.fit(X_weighted, y_cal)
        z_pcet_train = ridge_pcet.decision_function(X_weighted)
        z_pcet_test = ridge_pcet.decision_function(X_weighted_test)

        alpha = 1 / (1 + np.exp(-(z_pcet_train - z_gaze_train)))
        alpha_test = 1 / (1 + np.exp(-(z_pcet_test - z_gaze_test)))

        z_fused_train = alpha * z_pcet_train + (1 - alpha) * z_gaze_train
        z_fused_test = alpha_test * z_pcet_test + (1 - alpha_test) * z_gaze_test

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=seed)
        clf.fit(z_fused_train.reshape(-1, 1), y_cal)
        y_pred = clf.predict(z_fused_test.reshape(-1, 1))

    elif method == 'GBE_only':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_gaze_cal)
        X_test = scaler.transform(X_gaze_test)
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
        clf.fit(X_train, y_cal)
        y_pred = clf.predict(X_test)

    elif method == 'PCET+GBE_concat':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        pca_models = {}
        for c in [0, 1]:
            X_class = X_eeg_train[y_cal == c]
            if len(X_class) > 5:
                pca = PCA(n_components=min(5, len(X_class)-1))
                pca.fit(X_class)
                pca_models[c] = pca

        X_pcet = X_eeg_train
        X_pcet_test = X_eeg_test_s
        for c in pca_models:
            pca = pca_models[c]
            X_recon = pca.inverse_transform(pca.transform(X_eeg_train))
            X_pcet = np.hstack([X_pcet, np.abs(X_eeg_train - X_recon)])
            X_recon_t = pca.inverse_transform(pca.transform(X_eeg_test_s))
            X_pcet_test = np.hstack([X_pcet_test, np.abs(X_eeg_test_s - X_recon_t)])

        X_concat_train = np.hstack([X_pcet, X_gaze_train])
        X_concat_test = np.hstack([X_pcet_test, X_gaze_test_s])

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
        clf.fit(X_concat_train, y_cal)
        y_pred = clf.predict(X_concat_test)

    elif method == 'PCET+GBE_static_avg':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        pca_models = {}
        for c in [0, 1]:
            X_class = X_eeg_train[y_cal == c]
            if len(X_class) > 5:
                pca = PCA(n_components=min(5, len(X_class)-1))
                pca.fit(X_class)
                pca_models[c] = pca

        X_pcet = X_eeg_train
        X_pcet_test = X_eeg_test_s
        for c in pca_models:
            pca = pca_models[c]
            X_recon = pca.inverse_transform(pca.transform(X_eeg_train))
            X_pcet = np.hstack([X_pcet, np.abs(X_eeg_train - X_recon)])
            X_recon_t = pca.inverse_transform(pca.transform(X_eeg_test_s))
            X_pcet_test = np.hstack([X_pcet_test, np.abs(X_eeg_test_s - X_recon_t)])

        ridge_pcet = RidgeClassifier(alpha=0.1)
        ridge_pcet.fit(X_pcet, y_cal)
        z_pcet_train = ridge_pcet.decision_function(X_pcet)
        z_pcet_test = ridge_pcet.decision_function(X_pcet_test)

        ridge_gaze = RidgeClassifier(alpha=1.0)
        ridge_gaze.fit(X_gaze_train, y_cal)
        z_gaze_train = ridge_gaze.decision_function(X_gaze_train)
        z_gaze_test = ridge_gaze.decision_function(X_gaze_test_s)

        z_avg_train = 0.5 * z_pcet_train + 0.5 * z_gaze_train
        z_avg_test = 0.5 * z_pcet_test + 0.5 * z_gaze_test

        clf = RidgeClassifier(alpha=0.1)
        clf.fit(z_avg_train.reshape(-1, 1), y_cal)
        y_pred = clf.predict(z_avg_test.reshape(-1, 1))

    elif method == 'PCET+GBE+CAGF':
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train = scaler_eeg.fit_transform(X_eeg_cal)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        X_gaze_train = scaler_gaze.fit_transform(X_gaze_cal)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

        pca_models = {}
        for c in [0, 1]:
            X_class = X_eeg_train[y_cal == c]
            if len(X_class) > 5:
                pca = PCA(n_components=min(5, len(X_class)-1))
                pca.fit(X_class)
                pca_models[c] = pca

        X_pcet = X_eeg_train
        X_pcet_test = X_eeg_test_s
        for c in pca_models:
            pca = pca_models[c]
            X_recon = pca.inverse_transform(pca.transform(X_eeg_train))
            X_pcet = np.hstack([X_pcet, np.abs(X_eeg_train - X_recon)])
            X_recon_t = pca.inverse_transform(pca.transform(X_eeg_test_s))
            X_pcet_test = np.hstack([X_pcet_test, np.abs(X_eeg_test_s - X_recon_t)])

        ridge_pcet = RidgeClassifier(alpha=0.1)
        ridge_pcet.fit(X_pcet, y_cal)
        z_pcet_train = ridge_pcet.decision_function(X_pcet)
        z_pcet_test = ridge_pcet.decision_function(X_pcet_test)

        gbe_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
        gbe_mlp.fit(X_gaze_train, y_cal)
        z_gaze_train = gbe_mlp.predict_proba(X_gaze_train)[:, 1]
        z_gaze_test = gbe_mlp.predict_proba(X_gaze_test_s)[:, 1]

        alpha = 1 / (1 + np.exp(-(z_pcet_train - z_gaze_train)))
        alpha_test = 1 / (1 + np.exp(-(z_pcet_test - z_gaze_test)))

        z_fused_train = alpha * z_pcet_train + (1 - alpha) * z_gaze_train
        z_fused_test = alpha_test * z_pcet_test + (1 - alpha_test) * z_gaze_test

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=seed)
        clf.fit(z_fused_train.reshape(-1, 1), y_cal)
        y_pred = clf.predict(z_fused_test.reshape(-1, 1))

    else:
        raise ValueError(f"Unknown method: {method}")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)

    return {
        'subject': subject,
        'k': k,
        'seed': seed,
        'method': method,
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_acc': bacc,
        'n_cal': n_cal,
        'n_test': n_test
    }


if __name__ == '__main__':
    print("=" * 80)
    print("PCET+GBE+CAGF COMPLETE EXPERIMENT WITH CORRECTED ALIGNMENT")
    print("=" * 80)

    METHODS = [
        'EEG_SVM', 'EEG_MLP', 'Gaze_SVM', 'Gaze_MLP',
        'Raw_Fusion', 'Ridge_StaticAvg',
        'PCET_only', 'GETA_only', 'PCET+GETA+CAGF',
        'GBE_only', 'PCET+GBE_concat', 'PCET+GBE_static_avg', 'PCET+GBE+CAGF'
    ]

    all_results = []
    alignment_info = []

    for subject in SUBJECTS_16:
        print(f"\nLoading {subject}...")
        X_eeg, X_gaze, y, aligned_keys, info = load_aligned_eeg_gaze(subject)

        if X_eeg is None:
            print(f"  Skipping {subject} - files not found")
            continue

        label_consistency = info['label_consistency']

        assert len(X_eeg) == len(X_gaze) == len(y), \
            f"Length mismatch: EEG={len(X_eeg)}, Gaze={len(X_gaze)}, y={len(y)}"
        assert label_consistency == 1.0, \
            f"Label consistency is {label_consistency}, expected 1.0"

        print(f"  Aligned: {info['aligned_count']} samples")
        print(f"  EEG dim: {info['eeg_dim']}, Gaze dim: {info['gaze_dim']}")
        print(f"  NR: {info['nr_count']} ({info['nr_pct']:.1f}%), TSR: {info['tsr_count']}")
        print(f"  Label consistency: {label_consistency:.2%}")

        alignment_info.append(info)

        for k in [3, 5, 10, 20, 50]:
            for seed in range(5):
                for method in METHODS:
                    result = run_single_experiment(subject, X_eeg, X_gaze, y, k, seed, method)
                    all_results.append(result)

        print(f"  Completed k=3,5,10,20,50 x 5 seeds x {len(METHODS)} methods")

    df_results = pd.DataFrame(all_results)
    df_alignment = pd.DataFrame(alignment_info)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = df_results.groupby(['k', 'method']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_acc': ['mean', 'std']
    }).reset_index()

    print("\nAccuracy by method and k:")
    for k in [3, 5, 10, 20, 50]:
        print(f"\nk={k}:")
        k_results = summary[summary['k'] == k].copy()
        k_results.columns = ['_'.join(col).strip('_') for col in k_results.columns.values]
        k_results = k_results.sort_values('accuracy_mean', ascending=False)
        for _, row in k_results.iterrows():
            print(f"  {row['method']}: {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

    os.makedirs('results/final', exist_ok=True)
    os.makedirs('reports/final', exist_ok=True)

    df_results.to_csv('results/final/aligned_main_comparison.csv', index=False)
    df_alignment.to_csv('results/final/aligned_alignment_info.csv', index=False)

    print("\n" + "=" * 80)
    print("COMPARISON: GETA vs GBE")
    print("=" * 80)

    comparison_methods = ['GETA_only', 'GBE_only', 'PCET+GETA+CAGF', 'PCET+GBE+CAGF', 'PCET+GBE_concat', 'PCET+GBE_static_avg']
    comparison_df = df_results[df_results['method'].isin(comparison_methods)]

    comparison_summary = comparison_df.groupby(['k', 'method']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_acc': ['mean', 'std']
    }).reset_index()

    print("\nGETA vs GBE comparison:")
    for k in [3, 5, 10, 20, 50]:
        print(f"\nk={k}:")
        k_results = comparison_summary[comparison_summary['k'] == k].copy()
        k_results.columns = ['_'.join(col).strip('_') for col in k_results.columns.values]
        k_results = k_results.sort_values('accuracy_mean', ascending=False)
        for _, row in k_results.iterrows():
            print(f"  {row['method']}: {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

    comparison_df.to_csv('results/final/aligned_geta_vs_gbe_comparison.csv', index=False)

    print("\n\nFiles saved:")
    print("  - results/final/aligned_main_comparison.csv")
    print("  - results/final/aligned_alignment_info.csv")
    print("  - results/final/aligned_geta_vs_gbe_comparison.csv")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)