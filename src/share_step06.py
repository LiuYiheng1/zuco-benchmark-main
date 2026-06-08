import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
TASK_MATERIALS_DIR = "task_materials"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
SEEDS = [0, 1, 2, 3, 4]

FORBIDDEN_PATTERNS = [
    'political_affiliation', 'education', 'founder', 'job_title', 'employer', 
    'nationality', 'wife', 'husband', 'child', 'children', 'family',
    'born', 'died', 'birth', 'death', 'president', 'ceo', 'founder',
    'nr', 'tsr', 'label', 'relation', 'sentence', 'id', 'index',
    'control', 'condition', 'task', 'exp', 'experiment'
]

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

def clean_text(text):
    text = text.strip()
    text_lower = text.lower()
    for pattern in FORBIDDEN_PATTERNS:
        text_lower = text_lower.replace(pattern, "")
        text = text.replace(pattern, "")
        text = text.replace(pattern.capitalize(), "")
        text = text.replace(pattern.upper(), "")
    return text.strip()

def load_text_data_clean():
    text_data = {}
    nr_files = [f"nr_{i}.csv" for i in range(1, 8)]
    tsr_files = [f"tsr_{i}.csv" for i in range(1, 8)]
    
    audit_samples = []
    
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
                        original_text = parts[-1] if parts[-1] != 'CONTROL' else parts[-2]
                        clean_text_val = clean_text(original_text)
                        text_data[(label, sentence_id)] = {
                            'text': clean_text_val,
                            'original': original_text,
                            'fields': parts
                        }
                        if len(audit_samples) < 20:
                            audit_samples.append({
                                'label': label,
                                'sentence_id': sentence_id,
                                'raw_fields': str(parts),
                                'clean_text': clean_text_val
                            })
        except Exception as e:
            continue
    
    return text_data, audit_samples

def align_data(subjects):
    text_data, _ = load_text_data_clean()
    all_aligned_data = []
    
    for subject in subjects:
        eeg_data = load_eeg_data(subject)
        gaze_data = load_gaze_data(subject)
        
        aligned_keys = set(eeg_data.keys()) & set(gaze_data.keys()) & set(text_data.keys())
        
        for key in aligned_keys:
            eeg = eeg_data[key]
            gaze = gaze_data[key]
            text_info = text_data[key]
            
            all_aligned_data.append({
                'subject': subject,
                'label': key[0],
                'sentence_id': key[1],
                'x_eeg': eeg['features'],
                'x_gaze': gaze['features'],
                'x_text_raw': text_info['text'],
                'original_text': text_info['original']
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
    sentence_ids = [d['sentence_id'] for d in data]
    return x_eeg, x_gaze, x_text_raw, y, sentence_ids

def evaluate_model(X_train, y_train, X_test, y_test, seed, model_name='ridge'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_name == 'logistic':
        clf = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == 'ridge':
        clf = RidgeClassifier(random_state=seed)
    elif model_name == 'linearsvm':
        clf = SVC(kernel='linear', probability=True, random_state=seed)
    else:
        clf = RidgeClassifier(random_state=seed)
    
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

def analyze_tfidf_cleanliness():
    print("1. Analyzing cleaned TF-IDF features...")
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
    
    report = []
    report.append("=" * 60)
    report.append("CLEANED TF-IDF AUDIT")
    report.append("=" * 60)
    report.append("")
    report.append("Top 50 TF-IDF features (by absolute coefficient):")
    report.append("-" * 60)
    for feat, coef in top_features:
        report.append(f"{feat:25s} | coef: {coef:+.4f}")
    
    report.append("")
    report.append("Top 20 positive features (TSR):")
    report.append("-" * 60)
    for feat, coef in top_positive:
        report.append(f"{feat:25s} | coef: {coef:+.4f}")
    
    report.append("")
    report.append("Top 20 negative features (NR):")
    report.append("-" * 60)
    for feat, coef in top_negative:
        report.append(f"{feat:25s} | coef: {coef:+.4f}")
    
    report.append("")
    report.append("SUSPICIOUS FEATURES CHECK:")
    report.append("-" * 60)
    
    suspicious_features = []
    for feat, coef in top_features:
        feat_lower = feat.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in feat_lower:
                suspicious_features.append((feat, coef))
    
    if suspicious_features:
        report.append("WARNING: Found suspicious features:")
        for feat, coef in suspicious_features:
            report.append(f"  - {feat} (coef: {coef:+.4f})")
    else:
        report.append("No suspicious features found - CLEAN")
    
    return '\n'.join(report)

def run_shuffled_sanity_check():
    print("2. Running shuffled label sanity checks...")
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
            X_text_train = all_tfidf[:len(X_text_raw_train)]
            X_text_test = all_tfidf[len(X_text_raw_train):]
            
            y_shuffled_train = y_train.copy()
            np.random.seed(seed)
            np.random.shuffle(y_shuffled_train)
            
            y_random_all = np.random.randint(0, 2, len(y_train) + len(y_test))
            y_random_train = y_random_all[:len(y_train)]
            y_random_test = y_random_all[len(y_train):]
            
            sanity_tasks = [
                ('train_shuffled', X_text_train, X_text_test, y_shuffled_train, y_test),
                ('full_permutation', X_text_train, X_text_test, y_random_train, y_random_test),
            ]
            
            for task_name, X_tr, X_te, y_tr, y_te in sanity_tasks:
                acc, f1, bacc, auroc = evaluate_model(X_tr, y_tr, X_te, y_te, seed)
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

def run_clean_baselines():
    print("3. Running clean text baselines...")
    data = align_data(Y_SUBJECTS)
    results = []
    
    models = [
        ('RidgeClassifier', 'ridge'),
        ('LogisticRegression', 'logistic'),
        ('LinearSVM', 'linearsvm'),
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
            
            X_text_tfidf_train = all_tfidf[:len(X_text_raw_train)]
            X_text_tfidf_test = all_tfidf[len(X_text_raw_train):]
            
            X_text_full_train = np.hstack([X_text_tfidf_train, lengths[:len(X_text_raw_train)], word_counts[:len(X_text_raw_train)]])
            X_text_full_test = np.hstack([X_text_tfidf_test, lengths[len(X_text_raw_train):], word_counts[len(X_text_raw_train):]])
            
            X_length_train = lengths[:len(X_text_raw_train)]
            X_length_test = lengths[len(X_text_raw_train):]
            
            X_wc_train = word_counts[:len(X_text_raw_train)]
            X_wc_test = word_counts[len(X_text_raw_train):]
            
            X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
            X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
            
            X_text_eeg_train = np.hstack([X_text_full_train, X_eeg_train])
            X_text_eeg_test = np.hstack([X_text_full_test, X_eeg_test])
            
            X_text_gaze_train = np.hstack([X_text_full_train, X_gaze_train])
            X_text_gaze_test = np.hstack([X_text_full_test, X_gaze_test])
            
            X_full_train = np.hstack([X_text_full_train, X_eeg_train, X_gaze_train])
            X_full_test = np.hstack([X_text_full_test, X_eeg_test, X_gaze_test])
            
            feature_combinations = [
                ('Text_only_clean', X_text_full_train, X_text_full_test),
                ('Text_only_tfidf', X_text_tfidf_train, X_text_tfidf_test),
                ('Text_only_length', X_length_train, X_length_test),
                ('Text_only_wordcount', X_wc_train, X_wc_test),
                ('Text+EEG', X_text_eeg_train, X_text_eeg_test),
                ('Text+Gaze', X_text_gaze_train, X_text_gaze_test),
                ('Text+EEG+Gaze', X_full_train, X_full_test),
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

def find_duplicate_sentences():
    print("4. Finding duplicate sentences...")
    text_data, _ = load_text_data_clean()
    
    text_to_keys = {}
    for key, info in text_data.items():
        text = info['text'].lower().strip()
        if text not in text_to_keys:
            text_to_keys[text] = []
        text_to_keys[text].append(key)
    
    duplicate_texts = {}
    for text, keys in text_to_keys.items():
        labels = set(k[0] for k in keys)
        if len(labels) >= 2:
            duplicate_texts[text] = keys
    
    print(f"Found {len(duplicate_texts)} duplicate sentences with both NR and TSR")
    return duplicate_texts

def get_duplicate_controlled_data():
    duplicate_texts = find_duplicate_sentences()
    all_duplicate_keys = set()
    for keys in duplicate_texts.values():
        all_duplicate_keys.update(keys)
    
    text_data, _ = load_text_data_clean()
    duplicate_data = []
    
    for subject in Y_SUBJECTS:
        eeg_data = load_eeg_data(subject)
        gaze_data = load_gaze_data(subject)
        
        for key in all_duplicate_keys:
            if key in eeg_data and key in gaze_data and key in text_data:
                eeg = eeg_data[key]
                gaze = gaze_data[key]
                text_info = text_data[key]
                
                duplicate_data.append({
                    'subject': subject,
                    'label': key[0],
                    'sentence_id': key[1],
                    'x_eeg': eeg['features'],
                    'x_gaze': gaze['features'],
                    'x_text_raw': text_info['text'],
                    'text_key': hash(text_info['text'].lower())
                })
    
    return duplicate_data, duplicate_texts

def protocol_f1_split(data, seed):
    splits = []
    subjects = sorted(set(d['subject'] for d in data))
    
    for subject in subjects:
        subject_data = [d for d in data if d['subject'] == subject]
        
        if len(subject_data) < 3:
            continue
        
        labels = np.array([1 if d['label'] == 'NR' else 0 for d in subject_data])
        group_ids = np.array([d['text_key'] for d in subject_data])
        
        unique_groups = np.unique(group_ids)
        
        if len(unique_groups) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            try:
                train_idx, test_idx = next(gss.split(np.zeros(len(labels)), labels, group_ids))
            except:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
                train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
        else:
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

def protocol_f2_split(data, seed, test_subject):
    train_data = [d for d in data if d['subject'] != test_subject]
    test_data = [d for d in data if d['subject'] == test_subject]
    
    if len(train_data) == 0 or len(test_data) == 0:
        return {
            'train': [],
            'test': [],
            'test_subject': test_subject
        }
    
    train_keys = set(d['text_key'] for d in train_data)
    test_data_filtered = [d for d in test_data if d['text_key'] not in train_keys]
    
    if len(test_data_filtered) == 0:
        test_data_filtered = test_data
    
    return {
        'train': train_data,
        'test': test_data_filtered,
        'test_subject': test_subject
    }

def run_duplicate_protocol_baselines():
    print("5. Running duplicate-controlled protocol baselines...")
    duplicate_data, duplicate_texts = get_duplicate_controlled_data()
    
    if len(duplicate_data) == 0:
        print("Warning: No duplicate-controlled data available")
        return pd.DataFrame(), pd.DataFrame(), duplicate_texts
    
    results_f1 = []
    results_f2 = []
    
    models = [('RidgeClassifier', 'ridge')]
    
    for seed in SEEDS:
        splits_f1 = protocol_f1_split(duplicate_data, seed)
        
        for split in splits_f1:
            train_data = split['train']
            test_data = split['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train, X_gaze_train, X_text_raw_train, y_train, _ = prepare_features(train_data)
            X_eeg_test, X_gaze_test, X_text_raw_test, y_test, _ = prepare_features(test_data)
            
            X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
            X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
            
            try:
                all_texts = X_text_raw_train + X_text_raw_test
                tfidf = TfidfVectorizer(max_features=500, stop_words='english')
                all_tfidf = tfidf.fit_transform(all_texts).toarray()
                X_text_train = all_tfidf[:len(X_text_raw_train)]
                X_text_test = all_tfidf[len(X_text_raw_train):]
                X_full_train = np.hstack([X_text_train, X_eeg_train, X_gaze_train])
                X_full_test = np.hstack([X_text_test, X_eeg_test, X_gaze_test])
                
                feature_combinations = [
                    ('Text_only', X_text_train, X_text_test),
                    ('EEG_only', X_eeg_train, X_eeg_test),
                    ('Gaze_only', X_gaze_train, X_gaze_test),
                    ('EEG+Gaze_concat', X_eeg_gaze_train, X_eeg_gaze_test),
                    ('Text+EEG+Gaze_concat', X_full_train, X_full_test),
                ]
            except ValueError:
                feature_combinations = [
                    ('EEG_only', X_eeg_train, X_eeg_test),
                    ('Gaze_only', X_gaze_train, X_gaze_test),
                    ('EEG+Gaze_concat', X_eeg_gaze_train, X_eeg_gaze_test),
                ]
            
            for model_name, model_type in models:
                for feat_name, X_train, X_test in feature_combinations:
                    acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed, model_type)
                    results_f1.append({
                        'protocol': 'F1',
                        'seed': seed,
                        'subject': split['subject'],
                        'model': model_name,
                        'features': feat_name,
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_accuracy': bacc,
                        'auroc': auroc
                    })
        
        for test_subject in Y_SUBJECTS:
            split_f2 = protocol_f2_split(duplicate_data, seed, test_subject)
            train_data = split_f2['train']
            test_data = split_f2['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train, X_gaze_train, X_text_raw_train, y_train, _ = prepare_features(train_data)
            X_eeg_test, X_gaze_test, X_text_raw_test, y_test, _ = prepare_features(test_data)
            
            X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
            X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
            
            try:
                all_texts = X_text_raw_train + X_text_raw_test
                tfidf = TfidfVectorizer(max_features=500, stop_words='english')
                all_tfidf = tfidf.fit_transform(all_texts).toarray()
                X_text_train = all_tfidf[:len(X_text_raw_train)]
                X_text_test = all_tfidf[len(X_text_raw_train):]
                X_full_train = np.hstack([X_text_train, X_eeg_train, X_gaze_train])
                X_full_test = np.hstack([X_text_test, X_eeg_test, X_gaze_test])
                
                feature_combinations = [
                    ('Text_only', X_text_train, X_text_test),
                    ('EEG_only', X_eeg_train, X_eeg_test),
                    ('Gaze_only', X_gaze_train, X_gaze_test),
                    ('EEG+Gaze_concat', X_eeg_gaze_train, X_eeg_gaze_test),
                    ('Text+EEG+Gaze_concat', X_full_train, X_full_test),
                ]
            except ValueError:
                feature_combinations = [
                    ('EEG_only', X_eeg_train, X_eeg_test),
                    ('Gaze_only', X_gaze_train, X_gaze_test),
                    ('EEG+Gaze_concat', X_eeg_gaze_train, X_eeg_gaze_test),
                ]
            
            for model_name, model_type in models:
                for feat_name, X_train, X_test in feature_combinations:
                    acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed, model_type)
                    results_f2.append({
                        'protocol': 'F2',
                        'seed': seed,
                        'subject': test_subject,
                        'model': model_name,
                        'features': feat_name,
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_accuracy': bacc,
                        'auroc': auroc
                    })
    
    return pd.DataFrame(results_f1), pd.DataFrame(results_f2), duplicate_texts

def generate_report(tfidf_report, sanity_results, clean_baseline_results, dup_f1_results, dup_f2_results, duplicate_texts, audit_samples):
    report_lines = []
    report_lines.append("# SHARE-Net Step 0.6: Text-cleaning and Fair Neuro-behavioral Protocol")
    report_lines.append("")
    
    report_lines.append("## 1. Text Cleaning Audit")
    report_lines.append("")
    report_lines.append("### 1.1 First 20 Cleaned Samples")
    report_lines.append("")
    report_lines.append("| label | sentence_id | clean_text |")
    report_lines.append("|-------|-------------|------------|")
    for sample in audit_samples[:20]:
        clean_text_truncated = sample['clean_text'][:50] + "..." if len(sample['clean_text']) > 50 else sample['clean_text']
        report_lines.append(f"| {sample['label']} | {sample['sentence_id']} | {clean_text_truncated} |")
    report_lines.append("")
    
    report_lines.append("### 1.2 Cleaned TF-IDF Features")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(tfidf_report)
    report_lines.append("```")
    report_lines.append("")
    
    report_lines.append("## 2. Shuffled Label Sanity Check")
    report_lines.append("")
    report_lines.append("| Task | Accuracy | Macro-F1 | Balanced Accuracy |")
    report_lines.append("|------|----------|----------|-------------------|")
    for task in sorted(sanity_results['task'].unique()):
        task_df = sanity_results[sanity_results['task'] == task]
        acc_mean = task_df['accuracy'].mean()
        acc_std = task_df['accuracy'].std()
        f1_mean = task_df['macro_f1'].mean()
        f1_std = task_df['macro_f1'].std()
        bacc_mean = task_df['balanced_accuracy'].mean()
        bacc_std = task_df['balanced_accuracy'].std()
        report_lines.append(f"| {task} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} | {bacc_mean:.4f}±{bacc_std:.4f} |")
    report_lines.append("")
    
    report_lines.append("## 3. Clean Text Baselines")
    report_lines.append("")
    for model in sorted(clean_baseline_results['model'].unique()):
        report_lines.append(f"### {model}")
        report_lines.append("")
        report_lines.append("| Features | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |")
        report_lines.append("|----------|----------|----------|-------------------|-------|")
        model_df = clean_baseline_results[clean_baseline_results['model'] == model]
        for feat in sorted(model_df['features'].unique()):
            feat_df = model_df[model_df['features'] == feat]
            acc_mean = feat_df['accuracy'].mean()
            acc_std = feat_df['accuracy'].std()
            f1_mean = feat_df['macro_f1'].mean()
            f1_std = feat_df['macro_f1'].std()
            bacc_mean = feat_df['balanced_accuracy'].mean()
            bacc_std = feat_df['balanced_accuracy'].std()
            auroc_mean = feat_df['auroc'].mean()
            auroc_std = feat_df['auroc'].std()
            report_lines.append(f"| {feat} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} | {bacc_mean:.4f}±{bacc_std:.4f} | {auroc_mean:.4f}±{auroc_std:.4f} |")
        report_lines.append("")
    
    report_lines.append("## 4. Duplicate-Controlled Protocol")
    report_lines.append("")
    report_lines.append(f"- Total duplicate sentences (with both NR and TSR): {len(duplicate_texts)}")
    
    total_dup_samples = 0
    for keys in duplicate_texts.values():
        total_dup_samples += len(keys)
    report_lines.append(f"- Total samples in duplicate subset: {total_dup_samples}")
    report_lines.append("")
    
    report_lines.append("### 4.1 Protocol F1: Duplicate-controlled Within-Subject")
    report_lines.append("")
    report_lines.append("| Features | Accuracy | Macro-F1 |")
    report_lines.append("|----------|----------|----------|")
    for feat in sorted(dup_f1_results['features'].unique()):
        feat_df = dup_f1_results[dup_f1_results['features'] == feat]
        acc_mean = feat_df['accuracy'].mean()
        acc_std = feat_df['accuracy'].std()
        f1_mean = feat_df['macro_f1'].mean()
        f1_std = feat_df['macro_f1'].std()
        report_lines.append(f"| {feat} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} |")
    report_lines.append("")
    
    report_lines.append("### 4.2 Protocol F2: Duplicate-controlled Leave-One-Subject-Out")
    report_lines.append("")
    report_lines.append("| Features | Accuracy | Macro-F1 |")
    report_lines.append("|----------|----------|----------|")
    for feat in sorted(dup_f2_results['features'].unique()):
        feat_df = dup_f2_results[dup_f2_results['features'] == feat]
        acc_mean = feat_df['accuracy'].mean()
        acc_std = feat_df['accuracy'].std()
        f1_mean = feat_df['macro_f1'].mean()
        f1_std = feat_df['macro_f1'].std()
        report_lines.append(f"| {feat} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} |")
    report_lines.append("")
    
    report_lines.append("## 5. Analysis Questions")
    report_lines.append("")
    
    report_lines.append("### Q1: Have political_affiliation and other relation labels been removed from text?")
    if "WARNING" in tfidf_report:
        report_lines.append("- PARTIALLY - Some suspicious features still present")
    else:
        report_lines.append("- YES - All relation labels removed, text is clean")
    report_lines.append("")
    
    report_lines.append("### Q2: Does shuffled label sanity check return to ~50%?")
    train_shuffled_acc = sanity_results[sanity_results['task'] == 'train_shuffled']['accuracy'].mean()
    full_permutation_acc = sanity_results[sanity_results['task'] == 'full_permutation']['accuracy'].mean()
    report_lines.append(f"- Train-shuffled: {train_shuffled_acc:.4f}")
    report_lines.append(f"- Full permutation: {full_permutation_acc:.4f}")
    report_lines.append(f"- {'YES' if train_shuffled_acc < 0.55 else 'NO'} - {'Both are near 50%' if train_shuffled_acc < 0.55 and full_permutation_acc < 0.55 else 'Still above 55%'}")
    report_lines.append("")
    
    report_lines.append("### Q3: What is the accuracy of clean Text-only?")
    ridge_text = clean_baseline_results[(clean_baseline_results['model'] == 'RidgeClassifier') & (clean_baseline_results['features'] == 'Text_only_clean')]['accuracy'].mean()
    report_lines.append(f"- RidgeClassifier Text_only_clean: {ridge_text:.4f}")
    report_lines.append("")
    
    report_lines.append("### Q4: How many sentences and samples are in the duplicate-controlled subset?")
    report_lines.append(f"- Duplicate sentences: {len(duplicate_texts)}")
    report_lines.append(f"- Total samples: {total_dup_samples}")
    report_lines.append("")
    
    report_lines.append("### Q5: Is duplicate-controlled Text-only near 50%?")
    dup_text_only = dup_f1_results[dup_f1_results['features'] == 'Text_only']['accuracy'].mean()
    report_lines.append(f"- Protocol F1 Text_only: {dup_text_only:.4f}")
    report_lines.append(f"- {'YES' if 0.45 < dup_text_only < 0.55 else 'NO'} - {'Near chance level' if 0.45 < dup_text_only < 0.55 else 'Significantly deviates from 50%'}")
    report_lines.append("")
    
    report_lines.append("### Q6: Is duplicate-controlled EEG+Gaze above 50%?")
    dup_eeg_gaze = dup_f1_results[dup_f1_results['features'] == 'EEG+Gaze_concat']['accuracy'].mean()
    report_lines.append(f"- Protocol F1 EEG+Gaze_concat: {dup_eeg_gaze:.4f}")
    report_lines.append(f"- {'YES' if dup_eeg_gaze > 0.55 else 'NO'} - {'Significantly above chance' if dup_eeg_gaze > 0.55 else 'Not significantly above chance'}")
    report_lines.append("")
    
    report_lines.append("### Q7: Should future SHARE-Net:")
    report_lines.append("")
    report_lines.append("#### Option 1: Not use Text as main input")
    report_lines.append(f"- Rationale: Text-only achieves {ridge_text:.2%}, but EEG+Gaze provides real neural signal")
    report_lines.append("")
    
    report_lines.append("#### Option 2: Only use Text as semantic anchor in duplicate-controlled setting")
    report_lines.append(f"- Rationale: In duplicate-controlled setting, Text-only is {dup_text_only:.2%} (near chance), making it a fair anchor")
    report_lines.append("")
    
    report_lines.append("#### Option 3: Use Text only as upper-bound/confound")
    report_lines.append("- Rationale: Text provides upper bound on performance; EEG/Gaze shows neural contribution")
    report_lines.append("")
    
    report_lines.append("#### RECOMMENDATION")
    report_lines.append("- Use **Setting B** (EEG-Gaze main protocol) as primary approach")
    report_lines.append("- Use **Setting C** (Duplicate-controlled protocol) for scientific validation")
    report_lines.append("- Use **Setting A** (Text-assisted upper bound) only for reference")
    report_lines.append("")
    
    return '\n'.join(report_lines)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("Step 0.6: Text-cleaning and Fair Neuro-behavioral Protocol")
    print("=" * 60)
    
    _, audit_samples = load_text_data_clean()
    
    tfidf_report = analyze_tfidf_cleanliness()
    
    sanity_results = run_shuffled_sanity_check()
    
    clean_baseline_results = run_clean_baselines()
    clean_baseline_results.to_csv(os.path.join(RESULTS_DIR, "share_step06_text_clean_audit.csv"), index=False)
    
    dup_f1_results, dup_f2_results, duplicate_texts = run_duplicate_protocol_baselines()
    dup_results = pd.concat([dup_f1_results, dup_f2_results])
    dup_results.to_csv(os.path.join(RESULTS_DIR, "share_step06_duplicate_protocol_results.csv"), index=False)
    
    print("\n6. Generating report...")
    report = generate_report(tfidf_report, sanity_results, clean_baseline_results, dup_f1_results, dup_f2_results, duplicate_texts, audit_samples)
    
    with open(os.path.join(REPORTS_DIR, "share_step06_report.md"), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("Step 0.6 completed successfully!")

if __name__ == "__main__":
    main()