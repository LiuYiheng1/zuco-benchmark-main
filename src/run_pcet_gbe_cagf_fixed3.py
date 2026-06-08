import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS']

def load_subject_data(subject):
    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'
    
    if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
        return None, None, None
    
    eeg_feats = np.load(eeg_path, allow_pickle=True).item()
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()
    
    eeg_dict = {}
    for key in eeg_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            sentence_idx = int(parts[2])
            data = np.array(eeg_feats[key])
            # 移除最后一个元素（标签）
            if data[-1] in ['NR', 'TSR']:
                data = data[:-1]
            # 转换为float
            data = data.astype(float)
            eeg_dict[sentence_idx] = {'key': key, 'data': data, 'label': 0 if 'NR' in parts[1] else 1}
    
    gaze_dict = {}
    for key in gaze_feats.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            sentence_idx = int(parts[2])
            data = np.array(gaze_feats[key])
            # 移除最后一个元素（标签）
            if len(data) > 0 and data[-1] in ['NR', 'TSR']:
                data = data[:-1]
            # 转换为float
            data = data.astype(float)
            gaze_dict[sentence_idx] = {'key': key, 'data': data}
    
    common_indices = set(eeg_dict.keys()) & set(gaze_dict.keys())
    
    X_eeg = []
    X_gaze = []
    y = []
    
    for idx in common_indices:
        X_eeg.append(eeg_dict[idx]['data'])
        X_gaze.append(gaze_dict[idx]['data'])
        y.append(eeg_dict[idx]['label'])
    
    if len(y) == 0:
        return None, None, None
    
    return np.array(X_eeg), np.array(X_gaze), np.array(y)

results = []

for k in [3, 5, 10]:
    print(f"Processing k={k}")
    for seed in range(2):
        np.random.seed(seed)
        
        all_preds = {}
        all_y_true = []
        
        for subject in SUBJECTS[:4]:
            X_eeg, X_gaze, y = load_subject_data(subject)
            if X_eeg is None or len(y) < 20:
                print(f"Skip {subject}")
                continue
            
            print(f"Processing {subject}: {len(y)} samples, EEG dim: {X_eeg.shape[1]}, Gaze dim: {X_gaze.shape[1]}")
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            train_idx, test_idx = next(sss.split(X_eeg, y))
            
            X_eeg_cal, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
            X_gaze_cal, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
            y_cal, y_test = y[train_idx], y[test_idx]
            
            cal_idx = []
            for c in [0, 1]:
                c_idx = np.where(y_cal == c)[0]
                selected = np.random.choice(c_idx, min(k, len(c_idx)), replace=False)
                cal_idx.extend(selected)
            
            X_eeg_cal = X_eeg_cal[cal_idx]
            X_gaze_cal = X_gaze_cal[cal_idx]
            y_cal = y_cal[cal_idx]
            
            mlp_eeg = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
            mlp_eeg.fit(X_eeg_cal, y_cal)
            
            mlp_gaze = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
            mlp_gaze.fit(X_gaze_cal, y_cal)
            
            X_concat = np.hstack([X_eeg_cal, X_gaze_cal])
            mlp_concat = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
            mlp_concat.fit(X_concat, y_cal)
            
            scaler_pcet = StandardScaler()
            X_eeg_scaled = scaler_pcet.fit_transform(X_eeg_cal)
            
            pca_models = {}
            for c in [0, 1]:
                X_class = X_eeg_scaled[y_cal == c]
                if len(X_class) > 5:
                    pca = PCA(n_components=min(5, len(X_class)-1))
                    pca.fit(X_class)
                    pca_models[c] = pca
            
            X_pcet_cal = X_eeg_scaled
            for c in pca_models:
                pca = pca_models[c]
                X_recon = pca.inverse_transform(pca.transform(X_eeg_scaled))
                X_pcet_cal = np.hstack([X_pcet_cal, np.abs(X_eeg_scaled - X_recon)])
            
            ridge_pcet = RidgeClassifier(alpha=0.1)
            ridge_pcet.fit(X_pcet_cal, y_cal)
            
            X_eeg_test_scaled = scaler_pcet.transform(X_eeg_test)
            X_pcet_test = X_eeg_test_scaled
            for c in pca_models:
                pca = pca_models[c]
                X_recon = pca.inverse_transform(pca.transform(X_eeg_test_scaled))
                X_pcet_test = np.hstack([X_pcet_test, np.abs(X_eeg_test_scaled - X_recon)])
            
            scaler_gbe = StandardScaler()
            X_gaze_scaled = scaler_gbe.fit_transform(X_gaze_cal)
            gbe_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
            gbe_mlp.fit(X_gaze_scaled, y_cal)
            
            z_pcet_train = ridge_pcet.decision_function(X_pcet_cal)
            z_gaze_train = gbe_mlp.predict_proba(X_gaze_scaled)[:, 1]
            
            alpha = 1 / (1 + np.exp(-(z_pcet_train - z_gaze_train)))
            z_fused_train = alpha * z_pcet_train + (1 - alpha) * z_gaze_train
            
            final_mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=seed)
            final_mlp.fit(z_fused_train.reshape(-1, 1), y_cal)
            
            z_pcet_test = ridge_pcet.decision_function(X_pcet_test)
            z_gaze_test = gbe_mlp.predict_proba(scaler_gbe.transform(X_gaze_test))[:, 1]
            alpha_test = 1 / (1 + np.exp(-(z_pcet_test - z_gaze_test)))
            z_fused_test = alpha_test * z_pcet_test + (1 - alpha_test) * z_gaze_test
            
            preds = {
                'EEG_MLP': mlp_eeg.predict(X_eeg_test),
                'Gaze_MLP': mlp_gaze.predict(X_gaze_test),
                'Raw_Fusion': mlp_concat.predict(np.hstack([X_eeg_test, X_gaze_test])),
                'PCET_only': ridge_pcet.predict(X_pcet_test),
                'GBE_only': gbe_mlp.predict(scaler_gbe.transform(X_gaze_test)),
                'PCET+GBE+CAGF': final_mlp.predict(z_fused_test.reshape(-1, 1))
            }
            
            for method in preds:
                if method not in all_preds:
                    all_preds[method] = []
                all_preds[method].extend(preds[method])
            all_y_true.extend(y_test)
        
        if all_y_true:
            y_true = np.array(all_y_true)
            for method in all_preds:
                y_pred = np.array(all_preds[method])
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                bacc = balanced_accuracy_score(y_true, y_pred)
                
                results.append({
                    'k': k,
                    'seed': seed,
                    'method': method,
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_acc': bacc
                })
            print(f"Seed {seed}: {len(all_y_true)} test samples")

df = pd.DataFrame(results)
print("\nResults:")
print(df)

os.makedirs('results/final', exist_ok=True)
df.to_csv('results/final/pcet_gbe_cagf_results.csv', index=False)
print("\nSaved to results/final/pcet_gbe_cagf_results.csv")

summary = df.groupby(['k', 'method']).agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_acc': ['mean', 'std']
}).reset_index()

print("\nSummary:")
print(summary)

with open('reports/final/pcet_gbe_cagf_report.md', 'w') as f:
    f.write("# PCET + GBE + CAGF Experiment Report\n\n")
    f.write("## Overview\n\n")
    f.write("New model with GBE (Gaze Behavioral Encoding) instead of GETA.\n\n")
    f.write("## Key Changes from GETA version\n\n")
    f.write("- GBE directly uses gaze features\n")
    f.write("- No entropy/confidence calculation\n")
    f.write("- No gaze attention reweighting of EEG\n")
    f.write("- CAGF fusion remains the same\n\n")
    f.write("## Results\n\n")
    f.write(summary.to_markdown())

print("\nReport saved!")