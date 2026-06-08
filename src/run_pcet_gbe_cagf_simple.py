import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_subject_data(subject):
    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'
    
    if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
        return None, None, None
    
    eeg_feats = np.load(eeg_path, allow_pickle=True).item()
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()
    
    X_eeg = []
    X_gaze = []
    y = []
    
    for key in eeg_feats.keys():
        if key in gaze_feats:
            X_eeg.append(eeg_feats[key])
            X_gaze.append(gaze_feats[key])
            label = 0 if 'NR' in key else 1
            y.append(label)
    
    if len(y) == 0:
        return None, None, None
    
    return np.array(X_eeg), np.array(X_gaze), np.array(y)

class PCETModule:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca_models = {}
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        for c in np.unique(y):
            X_class = X_scaled[y == c]
            if len(X_class) > self.n_components:
                pca = PCA(n_components=self.n_components)
                pca.fit(X_class)
                self.pca_models[c] = pca
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        errors = []
        
        for c in sorted(self.pca_models.keys()):
            if c in self.pca_models:
                pca = self.pca_models[c]
                X_recon = pca.inverse_transform(pca.transform(X_scaled))
                errors.append(np.abs(X_scaled - X_recon))
        
        if errors:
            return np.hstack([X_scaled] + errors)
        return X_scaled

class GBEModule:
    def __init__(self):
        self.scaler = StandardScaler()
        self.gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    
    def fit(self, X_gaze, y):
        self.scaler.fit(X_gaze)
        self.gaze_mlp.fit(self.scaler.transform(X_gaze), y)
    
    def predict_proba(self, X_gaze):
        return self.gaze_mlp.predict_proba(self.scaler.transform(X_gaze))[:, 1]
    
    def predict(self, X_gaze):
        return self.gaze_mlp.predict(self.scaler.transform(X_gaze))

class CAGFModule:
    def __init__(self):
        self.final_mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
    
    def fit(self, z_pcet, z_gaze, y):
        alpha = 1 / (1 + np.exp(-(z_pcet - z_gaze)))
        z_fused = alpha * z_pcet + (1 - alpha) * z_gaze
        self.final_mlp.fit(z_fused.reshape(-1, 1), y)
    
    def predict(self, z_pcet, z_gaze):
        alpha = 1 / (1 + np.exp(-(z_pcet - z_gaze)))
        z_fused = alpha * z_pcet + (1 - alpha) * z_gaze
        return self.final_mlp.predict(z_fused.reshape(-1, 1))

results = []

for k in [3,5,10,20,50]:
    print(f"Processing k={k}")
    for seed in range(5):
        np.random.seed(seed)
        
        all_preds = {}
        all_y_true = []
        
        for subject in SUBJECTS:
            X_eeg, X_gaze, y = load_subject_data(subject)
            if X_eeg is None or len(y) < 10:
                continue
            
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
            
            # Baselines
            svm_eeg = SVC(kernel='linear', probability=True, random_state=seed)
            svm_eeg.fit(X_eeg_cal, y_cal)
            
            svm_gaze = SVC(kernel='linear', probability=True, random_state=seed)
            svm_gaze.fit(X_gaze_cal, y_cal)
            
            mlp_eeg = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
            mlp_eeg.fit(X_eeg_cal, y_cal)
            
            mlp_gaze = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
            mlp_gaze.fit(X_gaze_cal, y_cal)
            
            X_concat_cal = np.hstack([X_eeg_cal, X_gaze_cal])
            mlp_concat = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
            mlp_concat.fit(X_concat_cal, y_cal)
            
            ridge_eeg = RidgeClassifier(alpha=0.1)
            ridge_eeg.fit(X_eeg_cal, y_cal)
            
            ridge_gaze = RidgeClassifier(alpha=0.1)
            ridge_gaze.fit(X_gaze_cal, y_cal)
            
            # PCET
            pcet = PCETModule()
            pcet.fit(X_eeg_cal, y_cal)
            X_pcet_cal = pcet.transform(X_eeg_cal)
            X_pcet_test = pcet.transform(X_eeg_test)
            
            ridge_pcet = RidgeClassifier(alpha=0.1)
            ridge_pcet.fit(X_pcet_cal, y_cal)
            
            # GBE
            gbe = GBEModule()
            gbe.fit(X_gaze_cal, y_cal)
            
            # PCET+GBE_concat
            X_pg_concat_cal = np.hstack([X_pcet_cal, X_gaze_cal])
            mlp_pg_concat = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
            mlp_pg_concat.fit(X_pg_concat_cal, y_cal)
            
            # CAGF
            z_pcet_cal = ridge_pcet.decision_function(X_pcet_cal)
            z_gaze_cal = gbe.predict_proba(X_gaze_cal)
            
            cagf = CAGFModule()
            cagf.fit(z_pcet_cal, z_gaze_cal, y_cal)
            
            # Predict
            z_pcet_test = ridge_pcet.decision_function(X_pcet_test)
            z_gaze_test = gbe.predict_proba(X_gaze_test)
            
            preds = {
                'EEG_SVM': svm_eeg.predict(X_eeg_test),
                'Gaze_SVM': svm_gaze.predict(X_gaze_test),
                'EEG_MLP': mlp_eeg.predict(X_eeg_test),
                'Gaze_MLP': mlp_gaze.predict(X_gaze_test),
                'Raw_EEG_Gaze_Fusion': mlp_concat.predict(np.hstack([X_eeg_test, X_gaze_test])),
                'Ridge_StaticAvg': (ridge_eeg.decision_function(X_eeg_test) + ridge_gaze.decision_function(X_gaze_test)) > 0,
                'PCET_only': ridge_pcet.predict(X_pcet_test),
                'GBE_only': gbe.predict(X_gaze_test),
                'PCET+GBE_concat': mlp_pg_concat.predict(np.hstack([X_pcet_test, X_gaze_test])),
                'PCET+GBE_static_avg': (z_pcet_test + z_gaze_test) > 0,
                'PCET+GBE+CAGF': cagf.predict(z_pcet_test, z_gaze_test)
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

df = pd.DataFrame(results)
os.makedirs('results/final', exist_ok=True)
df.to_csv('results/final/pcet_gbe_cagf_results.csv', index=False)
print("Results saved!")

# Generate report
summary = df.groupby(['k', 'method']).agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'balanced_acc': ['mean', 'std']
}).reset_index()

print("\n=== PCET + GBE + CAGF Results ===")
print(summary.to_string())

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
    f.write("\n")

print("\nReport saved!")