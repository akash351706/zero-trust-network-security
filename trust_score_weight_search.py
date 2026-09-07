"""
Trust-Score Weight Search
--------------------------
Finds working values for w1, w2, w3 in the trust-score function:

    T(x) = 1 - [ w1 * y_hat(x) + w2 * sigmoid(s(x)) + w3 * D(x) ]

by grid-searching combinations that sum to 1, and picking whichever
combination gives the best F1 score on your validation set when the
resulting trust score is used to classify flows (T(x) < threshold -> malicious).

Run this on your own laptop (no EC2 needed) - it just needs your
already-trained models and your preprocessed validation split.

Requirements: pip install scikit-learn joblib numpy
"""

import numpy as np
import joblib
from sklearn.metrics import f1_score

# ---- 1. Load your already-trained models and validation data ----
# Adjust these paths to wherever your saved files actually are.
rf_model = joblib.load("random_forest_model.pkl")
if_model = joblib.load("isolation_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# X_val, y_val should be your held-out validation split.
# y_val should be binary: 0 = benign, 1 = malicious (collapse your 5 classes
# down to this binary view just for the weight search).
import pandas as pd
X_val = pd.read_csv("X_processed.csv")   # replace with your actual validation split
y_val_raw = pd.read_csv("y_processed.csv")
y_val = (y_val_raw.iloc[:, 0] != "Normal").astype(int).values  # 1 if any attack class

# ---- 2. Get the two base signals ----
# Random Forest posterior probability of the malicious class(es)
rf_probs = rf_model.predict_proba(X_val)
# Sum probability across all non-Normal classes if multi-class
if rf_probs.shape[1] > 2:
    normal_idx = list(rf_model.classes_).index("Normal")  # finds "Normal" wherever it actually is
    y_hat = 1 - rf_probs[:, normal_idx]
else:
    y_hat = rf_probs[:, 1]

# Isolation Forest anomaly score, normalised via sigmoid
raw_scores = -if_model.score_samples(X_val)  # higher = more anomalous
sigmoid_scores = 1 / (1 + np.exp(-raw_scores))

# Drift term D(x): until the Phase 2 drift monitor exists, use a placeholder
# of all zeros so the search still runs (w3 will naturally come out near 0).
D = np.zeros(len(X_val))

# ---- 3. Grid search over w1, w2, w3 (summing to 1) ----
best_f1 = -1
best_weights = None
best_threshold = None
step = 0.05

for w1 in np.arange(0, 1.0001, step):
    for w2 in np.arange(0, 1.0001 - w1, step):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue

        trust_score = 1 - (w1 * y_hat + w2 * sigmoid_scores + w3 * D)

        # Try a few thresholds on the trust score itself
        for threshold in np.arange(0.3, 0.71, 0.05):
            predicted_malicious = (trust_score < threshold).astype(int)
            f1 = f1_score(y_val, predicted_malicious)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))
                best_threshold = round(threshold, 2)

print(f"Best F1 score:        {best_f1:.4f}")
print(f"Best weights (w1,w2,w3): {best_weights}")
print(f"Best trust-score threshold: {best_threshold}")
print()
print("Use these weights in your trust-score function T(x) = "
      f"1 - [{best_weights[0]}*y_hat(x) + {best_weights[1]}*sigmoid(s(x)) + {best_weights[2]}*D(x)]")
print(f"Classify a flow as malicious when T(x) < {best_threshold}")
