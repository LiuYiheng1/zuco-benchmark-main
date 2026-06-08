#!/usr/bin/env python
"""
STAG-Read R2C: Subject Normalization + Robust Tabular Ensemble
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

OUTPUT_DIR = "results/r2c_norm_ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = "data/aligned_multimodal_y.npz"
METADATA_FILE = "data/aligned_multimodal_y_metadata.csv"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 
              'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']


def load_data():
    """加载 aligned 数据"""
    print("Loading aligned data...")
    data = np.load(DATA_FILE)
    meta = pd.read_csv(METADATA_FILE)
    
    print(f"  Keys: {list(data.keys())}")
    print(f"  Metadata columns: {meta.columns.tolist()}")
    
    X_eeg = data['eeg']
    X_gaze = data['gaze']
    y = data['y']
    subjects = meta['subject'].values
    
    X_concat = np.hstack([X_eeg, X_gaze])
    
    print(f"  EEG: {X_eeg.shape}, Gaze: {X_gaze.shape}")
    print(f"  Concat: {X_concat.shape}")
    print(f"  Labels: NR={np.sum(y==0)}, TSR={np.sum(y==1)}")
    print(f"  Unique subjects: {len(np.unique(subjects))}")
    
    return X_eeg, X_gaze, X_concat, y, subjects


def subject_normalization(X, method='none'):
    """Subject normalization variants"""
    if method == 'none':
        return X
    
    if method == 'N1_robust':
        scaler = RobustScaler()
        return scaler.fit_transform(X)
    
    if method == 'N2_quantile':
        scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
        return scaler.fit_transform(X)
    
    return X


def run_loso_experiment(X, y, subjects, model_name, model_factory, norm_method='none'):
    """运行 LOSO 实验"""
    results = []
    
    for test_subj in Y_SUBJECTS:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        X_train = subject_normalization(X_train, norm_method)
        X_test = subject_normalization(X_test, norm_method)
        
        try:
            clf = model_factory()
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            results.append({
                'test_subject': test_subj,
                'test_N': len(y_test),
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
        except Exception as e:
            print(f"    Error for {test_subj}: {e}")
    
    return pd.DataFrame(results)


def run_ensemble_experiment(X, y, subjects):
    """运行集成实验"""
    print("\n  Running Ensemble (ExtraTrees + LogReg)...")
    
    results = []
    
    for test_subj in Y_SUBJECTS:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf1 = ExtraTreesClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
        clf1.fit(X_train_scaled, y_train)
        prob1 = clf1.predict_proba(X_test_scaled)[:, 1]
        
        clf2 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf2.fit(X_train_scaled, y_train)
        prob2 = clf2.predict_proba(X_test_scaled)[:, 1]
        
        prob_avg = (prob1 + prob2) / 2
        y_pred = (prob_avg > 0.5).astype(int)
        
        results.append({
            'test_subject': test_subj,
            'test_N': len(y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'auroc': roc_auc_score(y_test, prob_avg)
        })
    
    return pd.DataFrame(results)


def main():
    print("="*70)
    print("STAG-Read R2C: Subject Normalization + Robust Ensemble")
    print("="*70)
    
    X_eeg, X_gaze, X_concat, y, subjects = load_data()
    
    model_factories = {
        'ExtraTrees_Balanced': lambda: ExtraTreesClassifier(n_estimators=300, max_depth=15, 
                                                            class_weight='balanced', random_state=42),
        'RandomForest_Balanced': lambda: RandomForestClassifier(n_estimators=300, max_depth=15, 
                                                               class_weight='balanced', random_state=42),
        'LogReg_Balanced': lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    }
    
    norm_methods = ['none', 'N1_robust', 'N2_quantile']
    
    normalization_results = []
    
    for norm in norm_methods:
        print(f"\n{'='*50}")
        print(f"Normalization: {norm}")
        print(f"{'='*50}")
        
        for model_name, model_factory in model_factories.items():
            print(f"\n  {model_name}...")
            
            results = run_loso_experiment(X_concat, y, subjects, model_name, model_factory, norm)
            
            if len(results) > 0:
                mean_f1 = results['macro_f1'].mean()
                std_f1 = results['macro_f1'].std()
                
                normalization_results.append({
                    'normalization': norm,
                    'model': model_name,
                    'mean_macro_f1': mean_f1,
                    'std_macro_f1': std_f1,
                    'mean_accuracy': results['accuracy'].mean(),
                    'mean_auroc': results['auroc'].mean(),
                    'n_subjects': len(results)
                })
                
                print(f"    Mean Macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    
    df_norm = pd.DataFrame(normalization_results)
    df_norm.to_csv(os.path.join(OUTPUT_DIR, "normalization_results.csv"), index=False)
    
    print("\n" + "="*70)
    print("Ensemble Experiment")
    print("="*70)
    
    df_ensemble = run_ensemble_experiment(X_concat, y, subjects)
    df_ensemble.to_csv(os.path.join(OUTPUT_DIR, "ensemble_results.csv"), index=False)
    
    if len(df_ensemble) > 0:
        print(f"\n  Ensemble Mean Macro-F1: {df_ensemble['macro_f1'].mean():.4f} +/- {df_ensemble['macro_f1'].std():.4f}")
    
    df_best = df_norm.loc[df_norm['mean_macro_f1'].idxmax()]
    
    print("\n" + "="*70)
    print("Best Configuration")
    print("="*70)
    print(f"  Normalization: {df_best['normalization']}")
    print(f"  Model: {df_best['model']}")
    print(f"  Mean Macro-F1: {df_best['mean_macro_f1']:.4f}")
    
    df_norm.to_csv(os.path.join(OUTPUT_DIR, "normalization_results.csv"), index=False)
    
    print("\n" + "="*70)
    print("R2C Complete!")
    print("="*70)
    print(f"\nOutputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()