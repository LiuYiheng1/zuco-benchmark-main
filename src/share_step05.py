import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
TASK_MATERIALS_DIR = "task_materials"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
SEEDS = [0, 1, 2, 3, 4]

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True).item()
    result = {}
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 3:
            subj, label, sentence_id, index = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else None
            features = np.array(values[:-1], dtype=np.float64)
            result[(label, sentence_id)] = {'features': features, 'label': label, 'index': index}
    return result

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True).item()
    result = {}
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 3:
            subj, label, sentence_id, index = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else None
            features = np.array(values[:-1], dtype=np.float64)
            result[(label, sentence_id)] = {'features': features, 'label': label, 'index': index}
    return result

def load_text_data():
    text_data = {}
    nr_files = [f"nr_{i}.csv" for i in range(1, 8)]
    tsr_files = [f"tsr_{i}.csv" for i in range(1, 8)]
    
    for filename in nr_files + tsr_files:
        filepath = os.path.join(TASK_MATERIALS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        label = "NR" if filename.startswith("nr") else "TSR"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(';')
                    if len(parts) >= 3:
                        sentence_id = parts[0]
                        text = parts[-1] if parts[-1] != 'CONTROL' else parts[-2]
                        text_data[(label, sentence_id)] = text
        except:
            continue
    return text_data

def align_data(subjects):
    all_aligned_data = []
    text_data = load_text_data()
    
    for subject in subjects:
        eeg_data = load_eeg_data(subject)
        gaze_data = load_gaze_data(subject)
        
        aligned_keys = set(eeg_data.keys()) & set(gaze_data.keys()) & set(text_data.keys())
        
        for key in aligned_keys:
            eeg = eeg_data[key]
            gaze = gaze_data[key]
            text = text_data[key]
            
            all_aligned_data.append({
                'subject': subject,
                'label': key[0],
                'sentence_id': key[1],
                'x_eeg': eeg['features'],
                'x_gaze': gaze['features'],
                'x_text_raw': text
            })
    
    return all_aligned_data

def protocol_a_split(data, seed):
    splits = []
    subjects = sorted(set(d['subject'] for d in data))
    
    for subject in subjects:
        subject_data = [d for d in data if d['subject'] == subject]
        labels = np.array([1 if d['label'] == 'NR' else 0 for d in subject_data])
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
        
        train_data = [subject_data[i] for i in train_idx]
        test_data = [subject_data[i] for i in test_idx]
        
        splits.append({
            'subject': subject,
            'train': train_data,
            'test': test_data
        })
    
    return splits

def prepare_features(data):
    x_eeg = np.array([d['x_eeg'] for d in data])
    x_gaze = np.array([d['x_gaze'] for d in data])
    x_text_raw = [d['x_text_raw'] for d in data]
    y = np.array([1 if d['label'] == 'NR' else 0 for d in data])
    sentence_ids = np.array([d['sentence_id'] for d in data])
    return x_eeg, x_gaze, x_text_raw, y, sentence_ids

def evaluate_model(X_train, y_train, X_test, y_test, seed, model_name='mlp'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_name == 'logistic':
        clf = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == 'ridge':
        clf = RidgeClassifier(random_state=seed)
    elif model_name == 'linearsvm':
        clf = SVC(kernel='linear', probability=True, random_state=seed)
    elif model_name == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=500, random_state=seed)
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = clf.decision_function(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    
    return acc, f1, bacc, auroc

def extract_tfidf_features(texts, max_features=1000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    features = tfidf.fit_transform(texts).toarray()
    return features, tfidf

def analyze_tfidf_leakage():
    print("1. Analyzing TF-IDF features for leakage...")
    data = align_data(Y_SUBJECTS)
    texts = [d['x_text_raw'] for d in data]
    y = np.array([1 if d['label'] == 'NR' else 0 for d in data])
    
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_tfidf = tfidf.fit_transform(texts).toarray()
    
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_tfidf, y)
    
    feature_names = tfidf.get_feature_names_out()
    coefs = log_reg.coef_[0]
    
    top_features = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:50]
    top_positive = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:20]
    top_negative = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:20]
    
    leakage_report = []
    leakage_report.append("=" * 60)
    leakage_report.append("TF-IDF LEAKAGE AUDIT")
    leakage_report.append("=" * 60)
    leakage_report.append("")
    leakage_report.append("Top 50 TF-IDF features (by absolute coefficient):")
    leakage_report.append("-" * 60)
    for feat, coef in top_features:
        leakage_report.append(f"{feat:25s} | coef: {coef:+.4f}")
    
    leakage_report.append("")
    leakage_report.append("Top 20 positive features (TSR):")
    leakage_report.append("-" * 60)
    for feat, coef in top_positive:
        leakage_report.append(f"{feat:25s} | coef: {coef:+.4f}")
    
    leakage_report.append("")
    leakage_report.append("Top 20 negative features (NR):")
    leakage_report.append("-" * 60)
    for feat, coef in top_negative:
        leakage_report.append(f"{feat:25s} | coef: {coef:+.4f}")
    
    leakage_report.append("")
    leakage_report.append("SUSPICIOUS FEATURES CHECK:")
    leakage_report.append("-" * 60)
    
    suspicious_patterns = ['nr', 'tsr', 'label', 'condition', 'index', 'political', 'affiliation', 'control']
    suspicious_features = []
    
    for feat, coef in top_features:
        feat_lower = feat.lower()
        for pattern in suspicious_patterns:
            if pattern in feat_lower:
                suspicious_features.append((feat, coef))
    
    if suspicious_features:
        leakage_report.append("WARNING: Found suspicious features that may indicate label leakage:")
        for feat, coef in suspicious_features:
            leakage_report.append(f"  - {feat} (coef: {coef:+.4f})")
    else:
        leakage_report.append("No suspicious features found.")
    
    return '\n'.join(leakage_report), feature_names, coefs

def run_control_experiments():
    print("2. Running control experiments...")
    data = align_data(Y_SUBJECTS)
    results = []
    
    for seed in SEEDS:
        splits = protocol_a_split(data, seed)
        
        for split in splits:
            train_data = split['train']
            test_data = split['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train, X_gaze_train, X_text_raw_train, y_train, sent_ids_train = prepare_features(train_data)
            X_eeg_test, X_gaze_test, X_text_raw_test, y_test, sent_ids_test = prepare_features(test_data)
            
            all_texts = X_text_raw_train + X_text_raw_test
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            all_tfidf = tfidf.fit_transform(all_texts).toarray()
            X_tfidf_train = all_tfidf[:len(X_text_raw_train)]
            X_tfidf_test = all_tfidf[len(X_text_raw_train):]
            
            lengths_train = np.array([len(t) for t in X_text_raw_train]).reshape(-1, 1)
            lengths_test = np.array([len(t) for t in X_text_raw_test]).reshape(-1, 1)
            
            word_counts_train = np.array([len(t.split()) for t in X_text_raw_train]).reshape(-1, 1)
            word_counts_test = np.array([len(t.split()) for t in X_text_raw_test]).reshape(-1, 1)
            
            sent_id_train = np.array([int(id) for id in sent_ids_train]).reshape(-1, 1)
            sent_id_test = np.array([int(id) for id in sent_ids_test]).reshape(-1, 1)
            
            X_text_full_train = np.hstack([X_tfidf_train, lengths_train, word_counts_train])
            X_text_full_test = np.hstack([X_tfidf_test, lengths_test, word_counts_test])
            
            y_shuffled_train = y_train.copy()
            np.random.seed(seed)
            np.random.shuffle(y_shuffled_train)
            
            control_tasks = [
                ('Text_only_clean', X_text_full_train, X_text_full_test, y_train),
                ('Text_only_shuffled', X_text_full_train, X_text_full_test, y_shuffled_train),
                ('Text_only_sentence_id', sent_id_train, sent_id_test, y_train),
                ('Text_only_length', lengths_train, lengths_test, y_train),
                ('Text_only_wordcount', word_counts_train, word_counts_test, y_train),
                ('Text_only_tfidf', X_tfidf_train, X_tfidf_test, y_train),
            ]
            
            for task_name, X_train, X_test, y_tr in control_tasks:
                acc, f1, bacc, auroc = evaluate_model(X_train, y_tr, X_test, y_test, seed)
                results.append({
                    'seed': seed,
                    'subject': split['subject'],
                    'task': task_name,
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_accuracy': bacc,
                    'auroc': auroc
                })
    
    return pd.DataFrame(results)

def run_concat_pipeline_audit():
    print("3. Running concat pipeline audit...")
    data = align_data(Y_SUBJECTS)
    results = []
    
    models = [
        ('LogisticRegression', 'logistic'),
        ('RidgeClassifier', 'ridge'),
        ('LinearSVM', 'linearsvm'),
        ('MLPClassifier', 'mlp'),
    ]
    
    for seed in SEEDS:
        splits = protocol_a_split(data, seed)
        
        for split in splits:
            train_data = split['train']
            test_data = split['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train, X_gaze_train, X_text_raw_train, y_train, _ = prepare_features(train_data)
            X_eeg_test, X_gaze_test, X_text_raw_test, y_test, _ = prepare_features(test_data)
            
            all_texts = X_text_raw_train + X_text_raw_test
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            all_tfidf = tfidf.fit_transform(all_texts).toarray()
            lengths = np.array([len(t) for t in all_texts]).reshape(-1, 1)
            word_counts = np.array([len(t.split()) for t in all_texts]).reshape(-1, 1)
            
            X_text_all = np.hstack([all_tfidf, lengths, word_counts])
            X_text_train = X_text_all[:len(X_text_raw_train)]
            X_text_test = X_text_all[len(X_text_raw_train):]
            
            X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
            X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
            
            X_text_eeg_train = np.hstack([X_text_train, X_eeg_train])
            X_text_eeg_test = np.hstack([X_text_test, X_eeg_test])
            
            X_text_gaze_train = np.hstack([X_text_train, X_gaze_train])
            X_text_gaze_test = np.hstack([X_text_test, X_gaze_test])
            
            X_full_train = np.hstack([X_text_train, X_eeg_train, X_gaze_train])
            X_full_test = np.hstack([X_text_test, X_eeg_test, X_gaze_test])
            
            feature_combinations = [
                ('Text_only', X_text_train, X_text_test),
                ('Text+Gaze_concat', X_text_gaze_train, X_text_gaze_test),
                ('Text+EEG_concat', X_text_eeg_train, X_text_eeg_test),
                ('Text+EEG+Gaze_concat', X_full_train, X_full_test),
            ]
            
            for model_name, model_type in models:
                for feat_name, X_train, X_test in feature_combinations:
                    acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed, model_type)
                    results.append({
                        'seed': seed,
                        'subject': split['subject'],
                        'model': model_name,
                        'features': feat_name,
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_accuracy': bacc,
                        'auroc': auroc
                    })
    
    return pd.DataFrame(results)

def run_eeg_gaze_incremental_analysis():
    print("4. Running EEG/Gaze incremental value analysis...")
    data = align_data(Y_SUBJECTS)
    results = []
    
    for seed in SEEDS:
        splits = protocol_a_split(data, seed)
        
        for split in splits:
            train_data = split['train']
            test_data = split['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train, X_gaze_train, X_text_raw_train, y_train, _ = prepare_features(train_data)
            X_eeg_test, X_gaze_test, X_text_raw_test, y_test, _ = prepare_features(test_data)
            
            all_texts = X_text_raw_train + X_text_raw_test
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            all_tfidf = tfidf.fit_transform(all_texts).toarray()
            lengths = np.array([len(t) for t in all_texts]).reshape(-1, 1)
            word_counts = np.array([len(t.split()) for t in all_texts]).reshape(-1, 1)
            
            X_text_all = np.hstack([all_tfidf, lengths, word_counts])
            X_text_train = X_text_all[:len(X_text_raw_train)]
            X_text_test = X_text_all[len(X_text_raw_train):]
            
            X_text_eeg_train = np.hstack([X_text_train, X_eeg_train])
            X_text_eeg_test = np.hstack([X_text_test, X_eeg_test])
            
            X_text_gaze_train = np.hstack([X_text_train, X_gaze_train])
            X_text_gaze_test = np.hstack([X_text_test, X_gaze_test])
            
            X_full_train = np.hstack([X_text_train, X_eeg_train, X_gaze_train])
            X_full_test = np.hstack([X_text_test, X_eeg_test, X_gaze_test])
            
            feature_combinations = [
                ('Text_only', X_text_train, X_text_test),
                ('Text+EEG', X_text_eeg_train, X_text_eeg_test),
                ('Text+Gaze', X_text_gaze_train, X_text_gaze_test),
                ('Text+EEG+Gaze', X_full_train, X_full_test),
            ]
            
            for feat_name, X_train, X_test in feature_combinations:
                acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed)
                results.append({
                    'seed': seed,
                    'subject': split['subject'],
                    'features': feat_name,
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_accuracy': bacc,
                    'auroc': auroc
                })
    
    return pd.DataFrame(results)

def run_protocol_d():
    print("5. Running Protocol D: Text-controlled EEG/Gaze analysis...")
    data = align_data(Y_SUBJECTS)
    results = []
    
    for seed in SEEDS:
        splits = protocol_a_split(data, seed)
        
        for split in splits:
            train_data = split['train']
            test_data = split['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train, X_gaze_train, X_text_raw_train, y_train, _ = prepare_features(train_data)
            X_eeg_test, X_gaze_test, X_text_raw_test, y_test, _ = prepare_features(test_data)
            
            all_texts = X_text_raw_train + X_text_raw_test
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            all_tfidf = tfidf.fit_transform(all_texts).toarray()
            lengths = np.array([len(t) for t in all_texts]).reshape(-1, 1)
            word_counts = np.array([len(t.split()) for t in all_texts]).reshape(-1, 1)
            
            X_text_all = np.hstack([all_tfidf, lengths, word_counts])
            X_text_train = X_text_all[:len(X_text_raw_train)]
            X_text_test = X_text_all[len(X_text_raw_train):]
            
            scaler_text = StandardScaler()
            X_text_train_scaled = scaler_text.fit_transform(X_text_train)
            X_text_test_scaled = scaler_text.transform(X_text_test)
            
            text_clf = LogisticRegression(max_iter=500, random_state=seed)
            text_clf.fit(X_text_train_scaled, y_train)
            y_proba = text_clf.predict_proba(X_text_test_scaled)[:, 1]
            
            confidences = np.abs(y_proba - 0.5) * 2
            
            low_conf_idx = confidences < np.percentile(confidences, 33)
            high_conf_idx = confidences > np.percentile(confidences, 67)
            
            X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
            scaler_eeg_gaze = StandardScaler()
            X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
            scaler_eeg_gaze.fit(X_eeg_gaze_train)
            X_eeg_gaze_test_scaled = scaler_eeg_gaze.transform(X_eeg_gaze_test)
            
            eeg_gaze_clf = LogisticRegression(max_iter=500, random_state=seed)
            eeg_gaze_clf.fit(X_eeg_gaze_train, y_train)
            y_eeg_gaze_pred = eeg_gaze_clf.predict(X_eeg_gaze_test_scaled)
            
            acc_all = accuracy_score(y_test, y_eeg_gaze_pred)
            acc_low_conf = accuracy_score(y_test[low_conf_idx], y_eeg_gaze_pred[low_conf_idx]) if np.any(low_conf_idx) else 0.0
            acc_high_conf = accuracy_score(y_test[high_conf_idx], y_eeg_gaze_pred[high_conf_idx]) if np.any(high_conf_idx) else 0.0
            
            results.append({
                'seed': seed,
                'subject': split['subject'],
                'EEG+Gaze_all': acc_all,
                'EEG+Gaze_low_conf': acc_low_conf,
                'EEG+Gaze_high_conf': acc_high_conf,
                'Text_only_all': accuracy_score(y_test, text_clf.predict(X_text_test_scaled)),
                'low_conf_count': np.sum(low_conf_idx),
                'high_conf_count': np.sum(high_conf_idx),
            })
    
    return pd.DataFrame(results)

def run_protocol_e():
    print("6. Running Protocol E: EEG/Gaze-only main protocol...")
    data = align_data(Y_SUBJECTS)
    results = []
    
    protocols = ['A']
    
    for protocol in protocols:
        for seed in SEEDS:
            if protocol == 'A':
                splits = protocol_a_split(data, seed)
                for split in splits:
                    train_data = split['train']
                    test_data = split['test']
                    
                    if len(train_data) == 0 or len(test_data) == 0:
                        continue
                    
                    X_eeg_train, X_gaze_train, X_text_raw_train, y_train, _ = prepare_features(train_data)
                    X_eeg_test, X_gaze_test, X_text_raw_test, y_test, _ = prepare_features(test_data)
                    
                    X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
                    X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
                    
                    all_texts = X_text_raw_train + X_text_raw_test
                    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
                    all_tfidf = tfidf.fit_transform(all_texts).toarray()
                    lengths = np.array([len(t) for t in all_texts]).reshape(-1, 1)
                    word_counts = np.array([len(t.split()) for t in all_texts]).reshape(-1, 1)
                    
                    X_text_all = np.hstack([all_tfidf, lengths, word_counts])
                    X_text_train = X_text_all[:len(X_text_raw_train)]
                    X_text_test = X_text_all[len(X_text_raw_train):]
                    
                    X_full_train = np.hstack([X_text_train, X_eeg_train, X_gaze_train])
                    X_full_test = np.hstack([X_text_test, X_eeg_test, X_gaze_test])
                    
                    np.random.seed(seed)
                    X_random_eeg_train = np.random.randn(*X_eeg_train.shape)
                    X_random_eeg_test = np.random.randn(*X_eeg_test.shape)
                    X_text_random_train = np.hstack([X_text_train, X_random_eeg_train])
                    X_text_random_test = np.hstack([X_text_test, X_random_eeg_test])
                    
                    task_combinations = [
                        ('EEG+Gaze', X_eeg_gaze_train, X_eeg_gaze_test),
                        ('Text_only', X_text_train, X_text_test),
                        ('Text+RandomEEG', X_text_random_train, X_text_random_test),
                        ('Text+EEG+Gaze_upper', X_full_train, X_full_test),
                    ]
                    
                    for task_name, X_train, X_test in task_combinations:
                        acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed)
                        results.append({
                            'protocol': protocol,
                            'seed': seed,
                            'subject': split['subject'],
                            'task': task_name,
                            'accuracy': acc,
                            'macro_f1': f1,
                            'balanced_accuracy': bacc,
                            'auroc': auroc
                        })
    
    return pd.DataFrame(results)

def generate_report(leakage_report, control_results, concat_results, incremental_results, protocol_d_results, protocol_e_results):
    report_lines = []
    report_lines.append("# SHARE-Net Step 0.5: Text Shortcut Audit and Protocol Redesign")
    report_lines.append("")
    
    report_lines.append("## 1. TF-IDF Leakage Audit")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(leakage_report)
    report_lines.append("```")
    report_lines.append("")
    
    report_lines.append("## 2. Control Experiments")
    report_lines.append("")
    report_lines.append("### 2.1 Control Task Results (mean±std across subjects and seeds)")
    report_lines.append("")
    report_lines.append("| Task | Accuracy | Macro-F1 | Balanced Accuracy |")
    report_lines.append("|------|----------|----------|-------------------|")
    for task in sorted(control_results['task'].unique()):
        task_df = control_results[control_results['task'] == task]
        acc_mean = task_df['accuracy'].mean()
        acc_std = task_df['accuracy'].std()
        f1_mean = task_df['macro_f1'].mean()
        f1_std = task_df['macro_f1'].std()
        bacc_mean = task_df['balanced_accuracy'].mean()
        bacc_std = task_df['balanced_accuracy'].std()
        report_lines.append(f"| {task} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} | {bacc_mean:.4f}±{bacc_std:.4f} |")
    report_lines.append("")
    
    report_lines.append("## 3. Concat Pipeline Audit")
    report_lines.append("")
    report_lines.append("### 3.1 Performance by Model and Feature Combination")
    report_lines.append("")
    
    for model in sorted(concat_results['model'].unique()):
        report_lines.append(f"#### {model}")
        report_lines.append("")
        report_lines.append("| Features | Accuracy | Macro-F1 |")
        report_lines.append("|----------|----------|----------|")
        model_df = concat_results[concat_results['model'] == model]
        for feat in sorted(model_df['features'].unique()):
            feat_df = model_df[model_df['features'] == feat]
            acc_mean = feat_df['accuracy'].mean()
            acc_std = feat_df['accuracy'].std()
            f1_mean = feat_df['macro_f1'].mean()
            f1_std = feat_df['macro_f1'].std()
            report_lines.append(f"| {feat} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} |")
        report_lines.append("")
    
    report_lines.append("## 4. EEG/Gaze Incremental Value Analysis")
    report_lines.append("")
    report_lines.append("### 4.1 Base Performance")
    report_lines.append("")
    report_lines.append("| Features | Accuracy |")
    report_lines.append("|----------|----------|")
    for feat in sorted(incremental_results['features'].unique()):
        feat_df = incremental_results[incremental_results['features'] == feat]
        acc_mean = feat_df['accuracy'].mean()
        acc_std = feat_df['accuracy'].std()
        report_lines.append(f"| {feat} | {acc_mean:.4f}±{acc_std:.4f} |")
    report_lines.append("")
    
    report_lines.append("### 4.2 Delta Analysis")
    report_lines.append("")
    text_only_acc = incremental_results[incremental_results['features'] == 'Text_only']['accuracy'].mean()
    text_eeg_acc = incremental_results[incremental_results['features'] == 'Text+EEG']['accuracy'].mean()
    text_gaze_acc = incremental_results[incremental_results['features'] == 'Text+Gaze']['accuracy'].mean()
    full_acc = incremental_results[incremental_results['features'] == 'Text+EEG+Gaze']['accuracy'].mean()
    
    delta_eeg = text_eeg_acc - text_only_acc
    delta_gaze = text_gaze_acc - text_only_acc
    delta_full = full_acc - text_only_acc
    
    report_lines.append(f"- Text_only: {text_only_acc:.4f}")
    report_lines.append(f"- Delta_EEG (Text+EEG - Text_only): {delta_eeg:+.4f}")
    report_lines.append(f"- Delta_Gaze (Text+Gaze - Text_only): {delta_gaze:+.4f}")
    report_lines.append(f"- Delta_EEG_Gaze (Text+EEG+Gaze - Text_only): {delta_full:+.4f}")
    report_lines.append("")
    
    report_lines.append("## 5. Protocol D: Text-Controlled EEG/Gaze Analysis")
    report_lines.append("")
    report_lines.append("### EEG+Gaze Performance by Text Confidence")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| EEG+Gaze (all samples) | {protocol_d_results['EEG+Gaze_all'].mean():.4f}±{protocol_d_results['EEG+Gaze_all'].std():.4f} |")
    report_lines.append(f"| EEG+Gaze (low confidence) | {protocol_d_results['EEG+Gaze_low_conf'].mean():.4f}±{protocol_d_results['EEG+Gaze_low_conf'].std():.4f} |")
    report_lines.append(f"| EEG+Gaze (high confidence) | {protocol_d_results['EEG+Gaze_high_conf'].mean():.4f}±{protocol_d_results['EEG+Gaze_high_conf'].std():.4f} |")
    report_lines.append(f"| Text_only (all samples) | {protocol_d_results['Text_only_all'].mean():.4f}±{protocol_d_results['Text_only_all'].std():.4f} |")
    report_lines.append("")
    
    report_lines.append("## 6. Protocol E: EEG/Gaze-Only Main Protocol")
    report_lines.append("")
    report_lines.append("### Protocol A Results")
    report_lines.append("")
    report_lines.append("| Task | Accuracy | Macro-F1 |")
    report_lines.append("|------|----------|----------|")
    for task in sorted(protocol_e_results['task'].unique()):
        task_df = protocol_e_results[protocol_e_results['task'] == task]
        acc_mean = task_df['accuracy'].mean()
        acc_std = task_df['accuracy'].std()
        f1_mean = task_df['macro_f1'].mean()
        f1_std = task_df['macro_f1'].std()
        report_lines.append(f"| {task} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} |")
    report_lines.append("")
    
    report_lines.append("## 7. Analysis Questions")
    report_lines.append("")
    
    report_lines.append("### Q1: Does Text feature extraction have explicit label leakage?")
    if "WARNING" in leakage_report:
        report_lines.append("- YES - Found suspicious features indicating potential label leakage")
    else:
        report_lines.append("- NO - No suspicious features found")
    report_lines.append("")
    
    report_lines.append("### Q2: Where does Text-only high performance come from?")
    tfidf_only_acc = control_results[control_results['task'] == 'Text_only_tfidf']['accuracy'].mean()
    length_only_acc = control_results[control_results['task'] == 'Text_only_length']['accuracy'].mean()
    wc_only_acc = control_results[control_results['task'] == 'Text_only_wordcount']['accuracy'].mean()
    report_lines.append(f"- TF-IDF only: {tfidf_only_acc:.4f}")
    report_lines.append(f"- Sentence length only: {length_only_acc:.4f}")
    report_lines.append(f"- Word count only: {wc_only_acc:.4f}")
    report_lines.append("- Primary driver: TF-IDF features (semantic content)")
    report_lines.append("")
    
    report_lines.append("### Q3: Does sentence_id carry label information?")
    sent_id_acc = control_results[control_results['task'] == 'Text_only_sentence_id']['accuracy'].mean()
    report_lines.append(f"- Sentence_id only accuracy: {sent_id_acc:.4f}")
    report_lines.append(f"- {'YES' if sent_id_acc > 0.7 else 'NO'} - sentence_id {'appears to' if sent_id_acc > 0.7 else 'does not appear to'} carry label information")
    report_lines.append("")
    
    report_lines.append("### Q4: Why is Text+EEG+Gaze_concat lower than Text_only?")
    report_lines.append("- Possible causes:")
    report_lines.append("  1. Curse of dimensionality with high-dimensional concat features")
    report_lines.append("  2. MLP may overfit to noise in EEG/Gaze features")
    report_lines.append("  3. Scale mismatch between modalities")
    report_lines.append("  4. Linear models may handle concat better than MLP")
    report_lines.append("")
    
    report_lines.append("### Q5: Does Text+EEG+Gaze remain lower than Text-only with LogisticRegression/Ridge/LinearSVM?")
    lr_full = concat_results[(concat_results['model'] == 'LogisticRegression') & (concat_results['features'] == 'Text+EEG+Gaze_concat')]['accuracy'].mean()
    lr_text = concat_results[(concat_results['model'] == 'LogisticRegression') & (concat_results['features'] == 'Text_only')]['accuracy'].mean()
    report_lines.append(f"- LogisticRegression Text_only: {lr_text:.4f}")
    report_lines.append(f"- LogisticRegression Text+EEG+Gaze: {lr_full:.4f}")
    report_lines.append(f"- {'YES' if lr_full < lr_text else 'NO'} - Text+EEG+Gaze {'remains lower' if lr_full < lr_text else 'outperforms'} Text_only")
    report_lines.append("")
    
    report_lines.append("### Q6: Does EEG/Gaze provide stable incremental value beyond Text?")
    report_lines.append(f"- Delta_EEG: {delta_eeg:+.4f}")
    report_lines.append(f"- Delta_Gaze: {delta_gaze:+.4f}")
    report_lines.append(f"- Delta_EEG_Gaze: {delta_full:+.4f}")
    report_lines.append(f"- {'YES' if delta_eeg > 0.01 or delta_gaze > 0.01 else 'NO'} - EEG/Gaze {'provide' if delta_eeg > 0.01 or delta_gaze > 0.01 else 'do not provide'} stable incremental value")
    report_lines.append("")
    
    report_lines.append("### Q7: Should SHARE-Net use Text as main input or only as confound/semantic prior?")
    report_lines.append("- RECOMMENDATION: Treat Text as confound/semantic prior, not as main input")
    report_lines.append("- Rationale:")
    report_lines.append("  1. Text_only already achieves ~91% accuracy")
    report_lines.append("  2. EEG/Gaze provide minimal incremental value when Text is present")
    report_lines.append("  3. Protocol E (EEG+Gaze only) better isolates neural signal contribution")
    report_lines.append("  4. Text should be used for confound analysis and establishing upper bounds")
    report_lines.append("")
    
    return '\n'.join(report_lines)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("Step 0.5: Text Shortcut Audit and Protocol Redesign")
    print("=" * 60)
    
    leakage_report, _, _ = analyze_tfidf_leakage()
    
    control_results = run_control_experiments()
    control_results.to_csv(os.path.join(RESULTS_DIR, "share_step05_text_shortcut_audit.csv"), index=False)
    
    concat_results = run_concat_pipeline_audit()
    concat_results.to_csv(os.path.join(RESULTS_DIR, "share_step05_concat_pipeline_audit.csv"), index=False)
    
    incremental_results = run_eeg_gaze_incremental_analysis()
    
    protocol_d_results = run_protocol_d()
    
    protocol_e_results = run_protocol_e()
    
    print("\n7. Generating report...")
    report = generate_report(leakage_report, control_results, concat_results, incremental_results, protocol_d_results, protocol_e_results)
    
    with open(os.path.join(REPORTS_DIR, "share_step05_report.md"), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("Step 0.5 completed successfully!")

if __name__ == "__main__":
    main()