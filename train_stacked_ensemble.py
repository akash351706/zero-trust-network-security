"""
Stacked Ensemble Meta-Learner
------------------------------
Combines three signals per flow into one final prediction:
  1. Random Forest posterior probability (already trained)
  2. Isolation Forest anomaly score (already trained)
  3. LSTM-Autoencoder reconstruction error (from train_lstm_autoencoder.py)

A simple Logistic Regression meta-learner is trained on these 3 inputs to
produce a final stacked prediction - this is what feeds y_hat(x)/s(x) in
the trust-score function from Section 2.

Run this AFTER train_lstm_autoencoder.py, in the same folder.

Requirements: pip install scikit-learn joblib numpy pandas
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

WINDOW_SIZE = 10  # must match train_lstm_autoencoder.py

# ---- 1. Load existing models ----
rf_model = joblib.load("random_forest_model.pkl")
if_model = joblib.load("isolation_forest_model.pkl")

# ---- 2. Load original flat data (for RF and IF, which use single flows) ----
# Note: RF and Isolation Forest were both trained/validated successfully on
# X_processed.csv AS-IS (unscaled) in the Section 2 weight-search script -
# tree-based models are scale-invariant, so this matches their training
# exactly. Only the LSTM (a neural network) needed scaling, which is applied
# separately inside train_lstm_autoencoder.py before it computes reconstruction
# errors - so no scaling is applied here.
X = pd.read_csv("X_processed.csv")
y_raw = pd.read_csv("y_processed.csv").iloc[:, 0]

# ---- 3. Load the LSTM reconstruction errors (already windowed/aligned) ----
lstm_results = pd.read_csv("lstm_reconstruction_errors.csv")

# The LSTM results have (len(X) - WINDOW_SIZE + 1) rows, each corresponding
# to the LAST flow in its window. Align RF/IF signals to the same rows.
offset = WINDOW_SIZE - 1
X_aligned = X.iloc[offset:].reset_index(drop=True)
y_aligned = y_raw.iloc[offset:].reset_index(drop=True)

assert len(X_aligned) == len(lstm_results), "Row count mismatch - check WINDOW_SIZE matches"

# ---- 4. Compute RF and IF signals on the aligned flows ----
rf_probs = rf_model.predict_proba(X_aligned)
normal_idx = list(rf_model.classes_).index("Normal")
rf_signal = 1 - rf_probs[:, normal_idx]  # probability of NOT normal

raw_if_scores = -if_model.score_samples(X_aligned)  # higher = more anomalous
if_signal = 1 / (1 + np.exp(-raw_if_scores))  # sigmoid-normalised, same as Section 2

lstm_signal = lstm_results["reconstruction_error"].values
# Normalise LSTM error onto a comparable 0-1 range
lstm_signal = (lstm_signal - lstm_signal.min()) / (lstm_signal.max() - lstm_signal.min() + 1e-9)

# ---- 5. Build the meta-learner's input matrix ----
meta_X = np.column_stack([rf_signal, if_signal, lstm_signal])
meta_y = (y_aligned != "Normal").astype(int).values  # binary: 0=Normal, 1=any attack

# ---- 6. Train/test split and train the meta-learner ----
X_train, X_test, y_train, y_test = train_test_split(
    meta_X, meta_y, test_size=0.2, stratify=meta_y, random_state=42
)

meta_learner = LogisticRegression(max_iter=1000, C=0.1)  # C=0.1 = stronger regularization,
# prevents the meta-learner from collapsing onto whichever single signal is
# strongest (Random Forest) and forces it to actually blend all three inputs
meta_learner.fit(X_train, y_train)

# ---- 7. Evaluate ----
y_pred = meta_learner.predict(X_test)
print("Stacked Ensemble - Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
print(f"F1 score: {f1_score(y_test, y_pred):.4f}")

# ---- 8. Save the meta-learner and its learned stacking weights ----
joblib.dump(meta_learner, "stacked_ensemble_meta_learner.pkl")

coef = meta_learner.coef_[0]
print("\nLearned stacking weights (higher = more influence on final decision):")
print(f"  Random Forest signal weight:  {coef[0]:.4f}")
print(f"  Isolation Forest signal weight: {coef[1]:.4f}")
print(f"  LSTM-AE signal weight:        {coef[2]:.4f}")
print(f"  Intercept: {meta_learner.intercept_[0]:.4f}")

print("\nSaved: stacked_ensemble_meta_learner.pkl")
print("This model + these weights feed y_hat(x)/s(x) in the Section 2 trust-score function.")
