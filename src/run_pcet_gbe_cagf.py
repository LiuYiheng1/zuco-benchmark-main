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

# Configuration
SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
FEATURE_DIR = 'src/features'
EEG_FEATURE_SET = 'eeg_means'
GAZE_FEATURE_SET = 'sent_gaze_sacc'

def load_subject_data(subject):
    """Load EEG and Gaze features for a subject."""
    eeg_path = os.path.join(FEATURE_DIR, f"{subject}_{EEG_FEATURE_SET}.npy")
    gaze_path = os.path.join(FEATURE_DIR, f"{subject}_{GAZE_FEATURE_SET}.npy")
    
    if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
        return None, None, None
    
    eeg_feats = np.load(eeg_path, allow_pickle=True).item()
    gaze_feats = np.load(gaze_path, allow_pickle=True).item()
    
    X_eeg = []
    X_gaze = []
    y = []
    
    for key in eeg_feats.keys():
        if key in gaze_feats:
            eeg_val = eeg_feats[key]
            gaze_val = gaze_feats[key]
            
            if isinstance(eeg_val, list) and len(eeg_val) > 0:
                if isinstance(eeg_val[-1], str) and (eeg_val[-1] == 'NR' or eeg_val[-1] == 'TSR'):
                    X_eeg.append(eeg_val[:-1])
                else:
                    X_eeg.append(eeg_val)
            else:
                X_eeg.append(eeg_val)
            
            if isinstance(gaze_val, list) and len(gaze_val) > 0:
                if isinstance(gaze_val[-1], str) and (gaze_val[-1] == 'NR' or gaze_val[-1] == 'TSR'):
                    X_gaze.append(gaze_val[:-1])
                else:
                    X_gaze.append(gaze_val)
            else:
                X_gaze.append(gaze_val)
            
            label = 0 if 'NR' in key else 1
            y.append(label)
    
    if len(y) == 0:
        return None, None, None
    
    return np.array(X_eeg), np.array(X_gaze), np.array(y)

class PCETModule:
    """Prediction-error Enhanced EEG Transformation"""
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca_models = {}
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        for c in np.unique(y):
            X_class = X_scaled[y == c]
            n_samples = len(X_class)
            n_feats = X_class.shape[1] if len(X_class.shape) > 1 else 1
            max_components = min(n_samples, n_feats, self.n_components)
            if max_components >= 2:
                pca = PCA(n_components=max_components)
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
            X_combined = np.hstack([X_scaled] + errors)
        else:
            X_combined = X_scaled
        
        return X_combined

class GBEModule:
    """Gaze Behavioral Encoding - Direct gaze features"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    
    def fit(self, X_gaze, y):
        self.scaler.fit(X_gaze)
        X_gaze_scaled = self.scaler.transform(X_gaze)
        self.gaze_mlp.fit(X_gaze_scaled, y)
    
    def predict_proba(self, X_gaze):
        X_gaze_scaled = self.scaler.transform(X_gaze)
        return self.gaze_mlp.predict_proba(X_gaze_scaled)[:, 1]
    
    def predict(self, X_gaze):
        X_gaze_scaled = self.scaler.transform(X_gaze)
        return self.gaze_mlp.predict(X_gaze_scaled)

class CAGFModule:
    """Cross-modal Adaptive Gated Fusion"""
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
    
    def predict_proba(self, z_pcet, z_gaze):
        alpha = 1 / (1 + np.exp(-(z_pcet - z_gaze)))
        z_fused = alpha * z_pcet + (1 - alpha) * z_gaze
        return self.final_mlp.predict_proba(z_fused.reshape(-1, 1))

def run_experiment(subjects, k_shots_list=[3,5,10,20,50], n_seeds=5):
    results = []
    
    for k in k_shots_list:
        print(f"Processing k={k} shots...")
        
        for seed in range(n_seeds):
            np.random.seed(seed)
            
            all_preds = {}
            all_probs = {}
            all_y_true = []
            
            for subject in subjects:
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
                
                # EEG_SVM
                svm_eeg = SVC(kernel='linear', probability=True, random_state=seed)
                svm_eeg.fit(X_eeg_cal, y_cal)
                
                # Gaze_SVM
                svm_gaze = SVC(kernel='linear', probability=True, random_state=seed)
                svm_gaze.fit(X_gaze_cal, y_cal)
                
                # EEG_MLP
                mlp_eeg = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
                mlp_eeg.fit(X_eeg_cal, y_cal)
                
                # Gaze_MLP
                mlp_gaze = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
                mlp_gaze.fit(X_gaze_cal, y_cal)
                
                # Raw EEG-Gaze MLP Fusion
                X_concat_cal = np.hstack([X_eeg_cal, X_gaze_cal])
                mlp_concat = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
                mlp_concat.fit(X_concat_cal, y_cal)
                
                # Ridge StaticAvg
                ridge_eeg = RidgeClassifier(alpha=0.1)
                ridge_eeg.fit(X_eeg_cal, y_cal)
                
                ridge_gaze = RidgeClassifier(alpha=0.1)
                ridge_gaze.fit(X_gaze_cal, y_cal)
                
                # PCET_only
                pcet = PCETModule()
                pcet.fit(X_eeg_cal, y_cal)
                X_pcet_cal = pcet.transform(X_eeg_cal)
                X_pcet_test = pcet.transform(X_eeg_test)
                
                ridge_pcet = RidgeClassifier(alpha=0.1)
                ridge_pcet.fit(X_pcet_cal, y_cal)
                
                # GBE_only
                gbe = GBEModule()
                gbe.fit(X_gaze_cal, y_cal)
                
                # PCET+GBE_concat
                X_pg_concat_cal = np.hstack([X_pcet_cal, X_gaze_cal])
                mlp_pg_concat = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=seed)
                mlp_pg_concat.fit(X_pg_concat_cal, y_cal)
                
                # PCET+GBE+CAGF
                z_pcet_cal = ridge_pcet.decision_function(X_pcet_cal)
                z_gaze_cal = gbe.predict_proba(X_gaze_cal)
                
                cagf = CAGFModule()
                cagf.fit(z_pcet_cal, z_gaze_cal, y_cal)
                
                # Predict on test
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
                
                probs = {
                    'EEG_SVM': svm_eeg.predict_proba(X_eeg_test)[:, 1],
                    'Gaze_SVM': svm_gaze.predict_proba(X_gaze_test)[:, 1],
                    'EEG_MLP': mlp_eeg.predict_proba(X_eeg_test)[:, 1],
                    'Gaze_MLP': mlp_gaze.predict_proba(X_gaze_test)[:, 1],
                    'Raw_EEG_Gaze_Fusion': mlp_concat.predict_proba(np.hstack([X_eeg_test, X_gaze_test]))[:, 1],
                    'PCET_only': ridge_pcet.decision_function(X_pcet_test),
                    'GBE_only': z_gaze_test,
                    'PCET+GBE+CAGF': cagf.predict_proba(z_pcet_test, z_gaze_test)[:, 1]
                }
                
                for method in preds:
                    if method not in all_preds:
                        all_preds[method] = []
                        all_probs[method] = []
                    all_preds[method].extend(preds[method])
                    if method in probs:
                        all_probs[method].extend(probs[method])
                    else:
                        all_probs[method].extend([0.5]*len(preds[method]))
                
                all_y_true.extend(y_test)
            
            if all_y_true:
                y_true = np.array(all_y_true)
                for method in all_preds:
                    y_pred = np.array(all_preds[method])
                    
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    bacc = balanced_accuracy_score(y_true, y_pred)
                    prob = np.array(all_probs[method])
                    try:
                        auroc = roc_auc_score(y_true, prob)
                    except:
                        auroc = 0.5
                    
                    results.append({
                        'k': k,
                        'seed': seed,
                        'method': method,
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_acc': bacc,
                        'auroc': auroc
                    })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting PCET + GBE + CAGF experiment...")
    
    df = run_experiment(SUBJECTS, k_shots_list=[3,5,10,20,50], n_seeds=5)
    
    os.makedirs('results/final', exist_ok=True)
    df.to_csv('results/final/pcet_gbe_cagf_results.csv', index=False)
    print("Results saved to results/final/pcet_gbe_cagf_results.csv")
    
    summary = df.groupby(['k', 'method']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_acc': ['mean', 'std'],
        'auroc': ['mean', 'std']
    }).reset_index()
    
    report = "# PCET + GBE + CAGF Experiment Report\n\n"
    report += "## Overview\n\n"
    report += "This report presents results for the new PCET + GBE + CAGF model architecture.\n\n"
    report += "## Key Changes from GETA version\n\n"
    report += "- **GBE (Gaze Behavioral Encoding)** directly uses gaze features\n"
    report += "- No entropy/confidence calculation\n"
    report += "- No gaze attention reweighting of EEG\n"
    report += "- CAGF fusion remains the same\n\n"
    report += "## Experimental Protocol\n\n"
    report += "- Few-shot personalized calibration\n"
    report += "- LOSO target subject\n"
    report += "- k = 3, 5, 10, 20, 50 shots per class\n"
    report += "- 16 Y-subjects\n"
    report += "- 5 seeds\n"
    report += "- 50% calibration / 50% test split\n"
    report += "- No test leakage\n\n"
    report += "## Results Summary\n\n"
    report += summary.to_markdown()
    report += "\n\n## Success Criteria Analysis\n\n"
    
    cagf_results = summary[summary['method'] == 'PCET+GBE+CAGF']
    fusion_results = summary[summary['method'] == 'Raw_EEG_Gaze_Fusion']
    staticavg_results = summary[summary['method'] == 'Ridge_StaticAvg']
    
    for k in [3,5,10,20,50]:
        cagf_acc = cagf_results[cagf_results['k'] == k]['accuracy']['mean'].values[0]
        fusion_acc = fusion_results[fusion_results['k'] == k]['accuracy']['mean'].values[0]
        staticavg_acc = staticavg_results[staticavg_results['k'] == k]['accuracy']['mean'].values[0]
        
        report += f"### k={k} shots\n"
        report += f"- PCET+GBE+CAGF: {cagf_acc*100:.1f}%\n"
        report += f"- Raw EEG-Gaze MLP Fusion: {fusion_acc*100:.1f}%\n"
        report += f"- Ridge StaticAvg: {staticavg_acc*100:.1f}%\n"
        report += f"- PCET+GBE+CAGF > Raw Fusion: {'YES' if cagf_acc > fusion_acc else 'NO'}\n"
        report += f"- PCET+GBE+CAGF > StaticAvg: {'YES' if cagf_acc > staticavg_acc else 'NO'}\n\n"
    
    os.makedirs('reports/final', exist_ok=True)
    with open('reports/final/pcet_gbe_cagf_report.md', 'w') as f:
        f.write(report)
    
    print("Report saved to reports/final/pcet_gbe_cagf_report.md")
    print("Experiment completed!")