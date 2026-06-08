"""Test sklearn import"""
import sys
log_file = open('d:/pycharmproject/zuco-benchmark-main/src/results/final/sklearn_test_log.txt', 'w')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

try:
    log("Starting...")
    import numpy as np
    log(f"numpy: {np.__version__}")

    from sklearn.preprocessing import StandardScaler
    log("StandardScaler imported")

    from sklearn.linear_model import LogisticRegression
    log("LogisticRegression imported")

    from sklearn.decomposition import PCA
    log("PCA imported")

    from sklearn.metrics import accuracy_score
    log("accuracy_score imported")

    log("All imports successful!")
    log_file.close()
except Exception as e:
    log(f"ERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())
    log_file.close()