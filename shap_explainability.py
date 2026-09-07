"""
SHAP Explainability
---------------------
Produces feature-importance plots for the Random Forest and the meta-learner,
showing WHICH features drive each classification decision. This is the
Week 11 deliverable from your roadmap.

Run AFTER train_stacked_ensemble.py, in the same folder.

Requirements: pip install shap matplotlib joblib pandas numpy
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

WINDOW_SIZE = 10

# ---- 1. Load models and data ----
rf_model = joblib.load("random_forest_model.pkl")
meta_learner = joblib.load("stacked_ensemble_meta_learner.pkl")

X = pd.read_csv("X_processed.csv")
y_raw = pd.read_csv("y_processed.csv").iloc[:, 0]

# Use a smaller sample for SHAP - it can be slow on large datasets
SAMPLE_SIZE = min(1000, len(X))
X_sample = X.sample(SAMPLE_SIZE, random_state=42)

# ---- 2. SHAP on the Random Forest ----
print("Computing SHAP values for Random Forest (this may take a minute)...")
rf_explainer = shap.TreeExplainer(rf_model)
shap_values_rf = rf_explainer.shap_values(X_sample)

# For multi-class RF, shap_values_rf is a list (one array per class) - use the
# "Normal" class's SHAP values, flipped, to show what drives "malicious" predictions
if isinstance(shap_values_rf, list):
    normal_idx = list(rf_model.classes_).index("Normal")
    shap_to_plot = -np.array(shap_values_rf[normal_idx])
else:
    shap_to_plot = shap_values_rf

plt.figure()
shap.summary_plot(shap_to_plot, X_sample, show=False)
plt.title("SHAP Feature Importance - Random Forest (driving 'malicious' prediction)")
plt.tight_layout()
plt.savefig("shap_random_forest_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_random_forest_summary.png")

# ---- 3. SHAP on the meta-learner (3 inputs: RF signal, IF signal, LSTM-AE signal) ----
# Rebuild the same 3-column input the meta-learner was trained on
if_model = joblib.load("isolation_forest_model.pkl")
lstm_results = pd.read_csv("lstm_reconstruction_errors.csv")

offset = WINDOW_SIZE - 1
X_aligned = X.iloc[offset:].reset_index(drop=True)

rf_probs_full = rf_model.predict_proba(X_aligned)
normal_idx = list(rf_model.classes_).index("Normal")
rf_signal_full = 1 - rf_probs_full[:, normal_idx]

raw_if_scores_full = -if_model.score_samples(X_aligned)
if_signal_full = 1 / (1 + np.exp(-raw_if_scores_full))

lstm_signal_full = lstm_results["reconstruction_error"].values
lstm_signal_full = (lstm_signal_full - lstm_signal_full.min()) / \
                    (lstm_signal_full.max() - lstm_signal_full.min() + 1e-9)

meta_X_full = pd.DataFrame({
    "RF_signal": rf_signal_full,
    "IF_signal": if_signal_full,
    "LSTM_AE_signal": lstm_signal_full,
})
meta_sample = meta_X_full.sample(min(1000, len(meta_X_full)), random_state=42)

print("Computing SHAP values for the meta-learner...")
meta_explainer = shap.LinearExplainer(meta_learner, meta_sample)
shap_values_meta = meta_explainer.shap_values(meta_sample)

plt.figure()
shap.summary_plot(shap_values_meta, meta_sample, show=False)
plt.title("SHAP Feature Importance - Stacked Ensemble Meta-Learner")
plt.tight_layout()
plt.savefig("shap_meta_learner_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_meta_learner_summary.png")

print("\nBoth SHAP plots saved. These show which features/signals drive each model's decisions.")
