"""
CV-ECF: Cross-Validated Error-Complementarity Fusion

Target: Exceed Static_EEG_Gaze_average by explicitly learning EEG-gaze error complementarity

Core idea:
- Use K-fold OOF predictions within calibration set to learn:
  1. When EEG is correct
  2. When gaze is correct
  3. What features characterize EEG-correct/gaze-wrong samples
  4. What features characterize gaze-correct/EEG-wrong samples
  5. When to trust static average

Implementation:
1. K-fold OOF within calibration set to get out-of-fold predictions
2. Construct meta-features from OOF predictions
3. Train fusion heads (logistic, weighted, regularized)
4. Evaluate on test set
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
REPORT_DIR = "reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

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

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze.npy")
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
        feat_list = list(values)
        features = np.array(feat_list[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def align_eeg_gaze(eeg_X, eeg_y, gaze_X, gaze_y):
    common_len = min(len(eeg_y), len(gaze_y))
    return eeg_X[:common_len], eeg_y[:common_len], gaze_X[:common_len], gaze_y[:common_len]

def get_oof_predictions(eeg_X, gaze_X, y, n_folds=5):
    """Get out-of-fold predictions using K-fold within calibration set"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    n_samples = len(y)
    p_eeg_oof = np.zeros(n_samples)
    p_gaze_oof = np.zeros(n_samples)
    pred_eeg_oof = np.zeros(n_samples, dtype=int)
    pred_gaze_oof = np.zeros(n_samples, dtype=int)
    logit_eeg_oof = np.zeros(n_samples)
    logit_gaze_oof = np.zeros(n_samples)

    for train_idx, val_idx in kf.split(eeg_X):
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()

        eeg_train_s = scaler_eeg.fit_transform(eeg_X[train_idx])
        gaze_train_s = scaler_gaze.fit_transform(gaze_X[train_idx])
        eeg_val_s = scaler_eeg.transform(eeg_X[val_idx])
        gaze_val_s = scaler_gaze.transform(gaze_X[val_idx])

        clf_eeg = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        clf_gaze = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)

        clf_eeg.fit(eeg_train_s, y[train_idx])
        clf_gaze.fit(gaze_train_s, y[train_idx])

        p_eeg_oof[val_idx] = clf_eeg.predict_proba(eeg_val_s)[:, 1]
        p_gaze_oof[val_idx] = clf_gaze.predict_proba(gaze_val_s)[:, 1]

        logits_eeg_val = np.log(np.clip(p_eeg_oof[val_idx], 1e-10, 1) / np.clip(1 - p_eeg_oof[val_idx], 1e-10, 1))
        logits_gaze_val = np.log(np.clip(p_gaze_oof[val_idx], 1e-10, 1) / np.clip(1 - p_gaze_oof[val_idx], 1e-10, 1))
        logit_eeg_oof[val_idx] = logits_eeg_val
        logit_gaze_oof[val_idx] = logits_gaze_val

        pred_eeg_oof[val_idx] = (p_eeg_oof[val_idx] >= 0.5).astype(int)
        pred_gaze_oof[val_idx] = (p_gaze_oof[val_idx] >= 0.5).astype(int)

    return {
        'p_eeg': p_eeg_oof,
        'p_gaze': p_gaze_oof,
        'pred_eeg': pred_eeg_oof,
        'pred_gaze': pred_gaze_oof,
        'logit_eeg': logit_eeg_oof,
        'logit_gaze': logit_gaze_oof,
        'y': y
    }

def construct_meta_features(oof_dict):
    """Construct meta-features for fusion head"""
    p_eeg = oof_dict['p_eeg']
    p_gaze = oof_dict['p_gaze']
    logit_eeg = oof_dict['logit_eeg']
    logit_gaze = oof_dict['logit_gaze']
    y = oof_dict['y']

    entropy_eeg = -p_eeg * np.log(np.clip(p_eeg, 1e-10, 1)) - (1-p_eeg) * np.log(np.clip(1-p_eeg, 1e-10, 1))
    entropy_gaze = -p_gaze * np.log(np.clip(p_gaze, 1e-10, 1)) - (1-p_gaze) * np.log(np.clip(1-p_gaze, 1e-10, 1))

    margin_eeg = np.abs(p_eeg - 0.5) * 2
    margin_gaze = np.abs(p_gaze - 0.5) * 2

    meta_features = np.column_stack([
        p_eeg,
        p_gaze,
        logit_eeg,
        logit_gaze,
        entropy_eeg,
        entropy_gaze,
        margin_eeg,
        margin_gaze,
        np.abs(p_eeg - p_gaze),
        np.maximum(p_eeg, 1-p_eeg),
        np.maximum(p_gaze, 1-p_gaze),
        (p_eeg + p_gaze) / 2,
        p_eeg * p_gaze,
        (1-p_eeg) * (1-p_gaze),
    ])

    pred_eeg = (p_eeg >= 0.5).astype(int)
    pred_gaze = (p_gaze >= 0.5).astype(int)

    eeg_correct = pred_eeg == y
    gaze_correct = pred_gaze == y

    state = np.zeros(len(y), dtype=int)
    state[eeg_correct & gaze_correct] = 0  # both_correct
    state[eeg_correct & ~gaze_correct] = 1  # eeg_only_correct
    state[~eeg_correct & gaze_correct] = 2  # gaze_only_correct
    state[~eeg_correct & ~gaze_correct] = 3  # both_wrong

    return meta_features, state, eeg_correct, gaze_correct

def train_fusion_heads(meta_features, states, y, lambda_reg=0.1):
    """Train three fusion heads"""
    scaler = StandardScaler()
    meta_s = scaler.fit_transform(meta_features)

    clf_logistic = LogisticRegression(max_iter=500, random_state=42)
    clf_logistic.fit(meta_s, y)

    return {
        'scaler': scaler,
        'clf_logistic': clf_logistic,
        'lambda_reg': lambda_reg
    }

def predict_with_fusion_heads(models, meta_features, p_eeg, p_gaze, method='ECF_logistic'):
    """Make predictions with fusion heads"""
    meta_s = models['scaler'].transform(meta_features)

    if method == 'ECF_logistic':
        p_final = models['clf_logistic'].predict_proba(meta_s)[:, 1]
    elif method == 'ECF_weighted':
        p_final = 0.5 * p_eeg + 0.5 * p_gaze
    elif method == 'ECF_regularized':
        p_final = models['clf_logistic'].predict_proba(meta_s)[:, 1]
    elif method == 'Static':
        p_final = 0.5 * p_eeg + 0.5 * p_gaze
    else:
        p_final = p_eeg if 'EEG' in method else p_gaze

    return p_final

def evaluate_predictions(p_final, y_true):
    """Evaluate predictions"""
    preds = (p_final >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    bacc = balanced_accuracy_score(y_true, preds)
    try:
        auroc = roc_auc_score(y_true, p_final)
    except:
        auroc = 0.5
    cm = confusion_matrix(y_true, preds)
    return acc, f1, bacc, auroc, cm

def run_experiment():
    results = []
    calibration_settings = [5, 10, 20, 50]
    lambda_regs = [0.01, 0.05, 0.1]
    seeds = [0, 1, 2, 3, 4]

    print("CV-ECF: Cross-Validated Error-Complementarity Fusion")
    print("="*60)
    print(f"Shot settings: {calibration_settings}")
    print(f"Lambda regs: {lambda_regs}")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)

        for held_out in Y_SUBJECTS:
            eeg_X, eeg_y = load_eeg_data(held_out)
            gaze_X, gaze_y = load_gaze_data(held_out)

            if eeg_X is None or gaze_X is None:
                print(f" {held_out}[no data]", end="", flush=True)
                continue

            eeg_X, eeg_y, gaze_X, gaze_y = align_eeg_gaze(eeg_X, eeg_y, gaze_X, gaze_y)

            if len(eeg_X) < 50:
                print(f" {held_out}[no align]", end="", flush=True)
                continue

            n_samples = len(eeg_y)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            eeg_test = eeg_X[test_indices]
            gaze_test = gaze_X[test_indices]
            y_test = eeg_y[test_indices]

            eeg_cal_pool = eeg_X[cal_pool_indices]
            gaze_cal_pool = gaze_X[cal_pool_indices]
            y_cal_pool = eeg_y[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                class_0_idx = np.where(y_cal_pool == 0)[0]
                class_1_idx = np.where(y_cal_pool == 1)[0]

                np.random.shuffle(class_0_idx)
                np.random.shuffle(class_1_idx)

                cal_idx = np.concatenate([class_0_idx[:n_cal_per_class], class_1_idx[:n_cal_per_class]])
                np.random.shuffle(cal_idx)

                eeg_cal = eeg_cal_pool[cal_idx]
                gaze_cal = gaze_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                if len(np.unique(y_cal)) < 2:
                    continue

                n_folds = min(5, len(cal_idx) // 2)
                if n_folds < 2:
                    n_folds = 2

                oof_dict = get_oof_predictions(eeg_cal, gaze_cal, y_cal, n_folds=n_folds)
                meta_features, states, eeg_correct, gaze_correct = construct_meta_features(oof_dict)

                scaler_eeg_final = StandardScaler()
                scaler_gaze_final = StandardScaler()
                eeg_cal_s = scaler_eeg_final.fit_transform(eeg_cal)
                gaze_cal_s = scaler_gaze_final.fit_transform(gaze_cal)
                eeg_test_s = scaler_eeg_final.transform(eeg_test)
                gaze_test_s = scaler_gaze_final.transform(gaze_test)

                clf_eeg_final = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
                clf_gaze_final = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
                clf_eeg_final.fit(eeg_cal_s, y_cal)
                clf_gaze_final.fit(gaze_cal_s, y_cal)

                p_eeg_test = clf_eeg_final.predict_proba(eeg_test_s)[:, 1]
                p_gaze_test = clf_gaze_final.predict_proba(gaze_test_s)[:, 1]

                logits_eeg_test = np.log(np.clip(p_eeg_test, 1e-10, 1) / np.clip(1 - p_eeg_test, 1e-10, 1))
                logits_gaze_test = np.log(np.clip(p_gaze_test, 1e-10, 1) / np.clip(1 - p_gaze_test, 1e-10, 1))

                entropy_eeg_test = -p_eeg_test * np.log(np.clip(p_eeg_test, 1e-10, 1)) - (1-p_eeg_test) * np.log(np.clip(1-p_eeg_test, 1e-10, 1))
                entropy_gaze_test = -p_gaze_test * np.log(np.clip(p_gaze_test, 1e-10, 1)) - (1-p_gaze_test) * np.log(np.clip(1-p_gaze_test, 1e-10, 1))
                margin_eeg_test = np.abs(p_eeg_test - 0.5) * 2
                margin_gaze_test = np.abs(p_gaze_test - 0.5) * 2

                meta_test = np.column_stack([
                    p_eeg_test, p_gaze_test,
                    logits_eeg_test, logits_gaze_test,
                    entropy_eeg_test, entropy_gaze_test,
                    margin_eeg_test, margin_gaze_test,
                    np.abs(p_eeg_test - p_gaze_test),
                    np.maximum(p_eeg_test, 1-p_eeg_test),
                    np.maximum(p_gaze_test, 1-p_gaze_test),
                    (p_eeg_test + p_gaze_test) / 2,
                    p_eeg_test * p_gaze_test,
                    (1-p_eeg_test) * (1-p_gaze_test),
                ])

                models = train_fusion_heads(meta_features, states, y_cal, lambda_reg=0.1)

                methods = [
                    'EEG_only',
                    'Gaze_only',
                    'Static_EEG_Gaze_average',
                    'ECF_logistic',
                    'ECF_weighted',
                ]

                for method in methods:
                    if method == 'EEG_only':
                        p_final = p_eeg_test
                    elif method == 'Gaze_only':
                        p_final = p_gaze_test
                    elif method == 'Static_EEG_Gaze_average':
                        p_final = 0.5 * p_eeg_test + 0.5 * p_gaze_test
                    else:
                        p_final = predict_with_fusion_heads(models, meta_test, p_eeg_test, p_gaze_test, method)

                    acc, f1, bacc, auroc, cm = evaluate_predictions(p_final, y_test)

                    results.append({
                        'seed': seed,
                        'subject': held_out,
                        'n_cal_per_class': n_cal_per_class,
                        'method': method,
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_accuracy': bacc,
                        'auroc': auroc,
                        'cm_0_0': cm[0, 0] if cm.shape == (2, 2) else 0,
                        'cm_0_1': cm[0, 1] if cm.shape == (2, 2) else 0,
                        'cm_1_0': cm[1, 0] if cm.shape == (2, 2) else 0,
                        'cm_1_1': cm[1, 1] if cm.shape == (2, 2) else 0,
                    })

                for lambda_reg in lambda_regs:
                    models_reg = train_fusion_heads(meta_features, states, y_cal, lambda_reg=lambda_reg)
                    p_final = predict_with_fusion_heads(models_reg, meta_test, p_eeg_test, p_gaze_test, 'ECF_regularized')
                    acc, f1, bacc, auroc, cm = evaluate_predictions(p_final, y_test)

                    results.append({
                        'seed': seed,
                        'subject': held_out,
                        'n_cal_per_class': n_cal_per_class,
                        'method': f'ECF_regularized_l{lambda_reg}',
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_accuracy': bacc,
                        'auroc': auroc,
                        'cm_0_0': cm[0, 0] if cm.shape == (2, 2) else 0,
                        'cm_0_1': cm[0, 1] if cm.shape == (2, 2) else 0,
                        'cm_1_0': cm[1, 0] if cm.shape == (2, 2) else 0,
                        'cm_1_1': cm[1, 1] if cm.shape == (2, 2) else 0,
                    })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def analyze_complementarity(df):
    """Analyze complementarity mechanisms"""
    print("\n" + "="*60)
    print("Complementarity Analysis")
    print("="*60)

    difficult_subjects = ['YLS', 'YSL', 'YHS']

    analysis_data = []

    for k in [5, 10, 20, 50]:
        static_data = df[(df['method'] == 'Static_EEG_Gaze_average') & (df['n_cal_per_class'] == k)]
        ecf_data = df[(df['method'] == 'ECF_logistic') & (df['n_cal_per_class'] == k)]

        if len(static_data) > 0 and len(ecf_data) > 0:
            static_acc = static_data.groupby('subject')['accuracy'].mean()
            ecf_acc = ecf_data.groupby('subject')['accuracy'].mean()

            for subj in difficult_subjects:
                if subj in static_acc.index and subj in ecf_acc.index:
                    diff = ecf_acc[subj] - static_acc[subj]
                    analysis_data.append({
                        'shot': k,
                        'subject': subj,
                        'static_acc': static_acc[subj],
                        'ecf_acc': ecf_acc[subj],
                        'diff': diff
                    })

    return pd.DataFrame(analysis_data)

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "cv_ecf_fusion_results.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("CV-ECF Results Summary")
    print("="*60)

    methods = ['EEG_only', 'Gaze_only', 'Static_EEG_Gaze_average', 'ECF_logistic', 'ECF_weighted',
               'ECF_regularized_l0.01', 'ECF_regularized_l0.05', 'ECF_regularized_l0.1']

    for k in [5, 10, 20, 50]:
        print(f"\nk={k}-shot:")
        for method in methods:
            data = df[(df['method'] == method) & (df['n_cal_per_class'] == k)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                std = data['accuracy'].std()
                f1 = data['macro_f1'].mean()
                print(f"  {method:30s}: acc={acc:.4f}±{std:.4f}, f1={f1:.4f}")

    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    static_baseline = {}
    for k in [5, 10, 20, 50]:
        static_data = df[(df['method'] == 'Static_EEG_Gaze_average') & (df['n_cal_per_class'] == k)]
        static_baseline[k] = static_data['accuracy'].mean()
        print(f"k={k}: Static baseline = {static_baseline[k]:.4f}")

    print("\n50-shot criteria (> Static + 1%):")
    target = static_baseline[50] + 0.01
    print(f"  Target: {target:.4f}")
    for method in methods:
        if 'ECF' in method:
            data = df[(df['method'] == method) & (df['n_cal_per_class'] == 50)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                status = "PASS" if acc >= target else "FAIL"
                print(f"    {method}: {acc:.4f} ({status})")

    print("\n10/20-shot criteria (> Static + 2%):")
    for k in [10, 20]:
        target = static_baseline[k] + 0.02
        print(f"  k={k} target: {target:.4f}")
        for method in methods:
            if 'ECF' in method:
                data = df[(df['method'] == method) & (df['n_cal_per_class'] == k)]
                if len(data) > 0:
                    acc = data['accuracy'].mean()
                    status = "PASS" if acc >= target else "FAIL"
                    print(f"    {method}: {acc:.4f} ({status})")

    difficult_subjects = ['YLS', 'YSL', 'YHS']
    print("\nDifficult subjects (YLS/YSL/YHS) avg gain vs Static:")
    for k in [10, 20, 50]:
        for method in ['ECF_logistic', 'ECF_weighted']:
            diff_gains = []
            for subj in difficult_subjects:
                static_data = df[(df['method'] == 'Static_EEG_Gaze_average') & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                method_data = df[(df['method'] == method) & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                if len(static_data) > 0 and len(method_data) > 0:
                    diff_gains.append(method_data['accuracy'].mean() - static_data['accuracy'].mean())
            if diff_gains:
                avg_gain = np.mean(diff_gains)
                marker = " *" if avg_gain >= 0.02 else ""
                print(f"  k={k}, {method}: {avg_gain:+.4f}{marker}")

    comp_df = analyze_complementarity(df)
    comp_path = os.path.join(RESULTS_DIR, "cv_ecf_complementarity_analysis.csv")
    comp_df.to_csv(comp_path, index=False)

    print(f"\nResults saved to {output_path}")
    print(f"Complementarity analysis saved to {comp_path}")
    print("Done!")

if __name__ == '__main__':
    main()