"""
Statistical Significance Testing + Baseline Comparison
---------------------------------------------------------
Answers the question: is the stacked ensemble ACTUALLY better than Random
Forest alone, or just tied? A single F1 number can't answer this - we need
per-fold scores and a paired significance test.

Also builds the 3-paper baseline comparison table using the papers from
your Section 1 literature review.

Run AFTER train_stacked_ensemble.py, in the same folder.

Requirements: pip install scikit-learn scipy numpy pandas joblib
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy import stats

WINDOW_SIZE = 10

# ---- 1. Rebuild the same aligned signals as train_stacked_ensemble.py ----
rf_model = joblib.load("random_forest_model.pkl")
if_model = joblib.load("isolation_forest_model.pkl")

X = pd.read_csv("X_processed.csv")
y_raw = pd.read_csv("y_processed.csv").iloc[:, 0]

lstm_results = pd.read_csv("lstm_reconstruction_errors.csv")
offset = WINDOW_SIZE - 1
X_aligned = X.iloc[offset:].reset_index(drop=True)
y_aligned = y_raw.iloc[offset:].reset_index(drop=True)

rf_probs = rf_model.predict_proba(X_aligned)
normal_idx = list(rf_model.classes_).index("Normal")
rf_signal = 1 - rf_probs[:, normal_idx]

raw_if_scores = -if_model.score_samples(X_aligned)
if_signal = 1 / (1 + np.exp(-raw_if_scores))

lstm_signal = lstm_results["reconstruction_error"].values
lstm_signal = (lstm_signal - lstm_signal.min()) / (lstm_signal.max() - lstm_signal.min() + 1e-9)

meta_X = np.column_stack([rf_signal, if_signal, lstm_signal])
meta_y = (y_aligned != "Normal").astype(int).values

# ---- 2. 5-fold CV: compare RF-alone vs Isolation-Forest-alone vs Ensemble ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_alone_f1_per_fold = []
if_alone_f1_per_fold = []
ensemble_f1_per_fold = []

for fold_num, (train_idx, test_idx) in enumerate(skf.split(meta_X, meta_y), start=1):
    y_test = meta_y[test_idx]

    # RF alone: threshold its own signal at 0.5
    rf_pred = (rf_signal[test_idx] > 0.5).astype(int)
    rf_alone_f1_per_fold.append(f1_score(y_test, rf_pred))

    # Isolation Forest alone: threshold at 0.5 too
    if_pred = (if_signal[test_idx] > 0.5).astype(int)
    if_alone_f1_per_fold.append(f1_score(y_test, if_pred))

    # Ensemble: train a fresh meta-learner on this fold's train split, test on test split
    fold_meta = LogisticRegression(max_iter=1000, C=0.1)
    fold_meta.fit(meta_X[train_idx], meta_y[train_idx])
    ens_pred = fold_meta.predict(meta_X[test_idx])
    ensemble_f1_per_fold.append(f1_score(y_test, ens_pred))

    print(f"Fold {fold_num}: RF-alone F1={rf_alone_f1_per_fold[-1]:.4f}  "
          f"IF-alone F1={if_alone_f1_per_fold[-1]:.4f}  "
          f"Ensemble F1={ensemble_f1_per_fold[-1]:.4f}")

print(f"\nMean +/- std across 5 folds:")
print(f"  RF alone:   {np.mean(rf_alone_f1_per_fold):.4f} +/- {np.std(rf_alone_f1_per_fold):.4f}")
print(f"  IF alone:   {np.mean(if_alone_f1_per_fold):.4f} +/- {np.std(if_alone_f1_per_fold):.4f}")
print(f"  Ensemble:   {np.mean(ensemble_f1_per_fold):.4f} +/- {np.std(ensemble_f1_per_fold):.4f}")

# ---- 3. Paired significance testing: Ensemble vs each base learner ----
print("\n--- Paired Significance Tests (Ensemble vs base learners) ---")

t_stat_rf, p_val_rf = stats.ttest_rel(ensemble_f1_per_fold, rf_alone_f1_per_fold)
print(f"Ensemble vs RF alone:  paired t-test  t={t_stat_rf:.4f}  p={p_val_rf:.4f}"
      f"  -> {'SIGNIFICANT (p<0.05)' if p_val_rf < 0.05 else 'not significant at p<0.05'}")

t_stat_if, p_val_if = stats.ttest_rel(ensemble_f1_per_fold, if_alone_f1_per_fold)
print(f"Ensemble vs IF alone:  paired t-test  t={t_stat_if:.4f}  p={p_val_if:.4f}"
      f"  -> {'SIGNIFICANT (p<0.05)' if p_val_if < 0.05 else 'not significant at p<0.05'}")

# Wilcoxon signed-rank as a non-parametric backup (more robust with only 5 folds)
try:
    w_stat_rf, wp_val_rf = stats.wilcoxon(ensemble_f1_per_fold, rf_alone_f1_per_fold)
    print(f"Ensemble vs RF alone:  Wilcoxon signed-rank  W={w_stat_rf:.4f}  p={wp_val_rf:.4f}")
except ValueError as e:
    print(f"Wilcoxon vs RF alone skipped: {e} (needs some folds to differ)")

# ---- 4. Baseline comparison table (from Section 1 literature review) ----
print("\n--- 3-Paper Baseline Comparison Table (NSL-KDD) ---")
baseline_table = pd.DataFrame([
    {"Source": "Roy et al. (stacking ensemble)", "Reported Accuracy/F1": "98.5% / F1 not reported separately"},
    {"Source": "De Souza et al. (two-step ensemble)", "Reported Accuracy/F1": "99.81% accuracy"},
    {"Source": "Scientific Reports FA-CNN+Autoencoder", "Reported Accuracy/F1": "97% (NSL-KDD)"},
    {"Source": "This Project (stacked IF+RF+LSTM-AE)",
     "Reported Accuracy/F1": f"F1={np.mean(ensemble_f1_per_fold):.4f} (5-fold mean)"},
])
print(baseline_table.to_string(index=False))
baseline_table.to_csv("baseline_comparison_table.csv", index=False)

# ---- 5. Save the fold-level results for your report ----
fold_results = pd.DataFrame({
    "fold": range(1, 6),
    "rf_alone_f1": rf_alone_f1_per_fold,
    "if_alone_f1": if_alone_f1_per_fold,
    "ensemble_f1": ensemble_f1_per_fold,
})
fold_results.to_csv("significance_test_fold_results.csv", index=False)
print("\nSaved: baseline_comparison_table.csv, significance_test_fold_results.csv")
