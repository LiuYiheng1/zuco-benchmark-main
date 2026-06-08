import os
import sys

log_path = 'd:/pycharmproject/zuco-benchmark-main/src/results/final/sci_log.txt'
log_file = open(log_path, 'w')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

log("Step 1: Starting")
log(f"CWD: {os.getcwd()}")

try:
    os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
    log(f"Step 2: CWD changed to {os.getcwd()}")
except Exception as e:
    log(f"Step 2 ERROR: {e}")
    log_file.close()
    sys.exit(1)

try:
    import numpy as np
    log(f"Step 3: numpy imported: {np.__version__}")
except Exception as e:
    log(f"Step 3 ERROR: {e}")
    log_file.close()
    sys.exit(1)

try:
    import pandas as pd
    log(f"Step 4: pandas imported: {pd.__version__}")
except Exception as e:
    log(f"Step 4 ERROR: {e}")
    log_file.close()
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    log("Step 5: StandardScaler imported")
except Exception as e:
    log(f"Step 5 ERROR: {e}")
    log_file.close()
    sys.exit(1)

try:
    from sklearn.linear_model import LogisticRegression
    log("Step 6: LogisticRegression imported")
except Exception as e:
    log(f"Step 6 ERROR: {e}")
    log_file.close()
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    log("Step 7: PCA imported")
except Exception as e:
    log(f"Step 7 ERROR: {e}")
    log_file.close()
    sys.exit(1)

try:
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
    log("Step 8: sklearn.metrics imported")
except Exception as e:
    log(f"Step 8 ERROR: {e}")
    log_file.close()
    sys.exit(1)

log("Step 9: All imports successful!")
log_file.close()
print("Script completed successfully!")