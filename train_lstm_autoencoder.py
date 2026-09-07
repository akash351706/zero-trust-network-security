"""
LSTM-Autoencoder Temporal Baseline
-----------------------------------
Trains an LSTM-autoencoder on sequences of BENIGN (Normal) flows only, then
uses reconstruction error as an anomaly signal - the same idea as Isolation
Forest, but capturing temporal/sequential patterns across consecutive flows
rather than treating each flow independently.

Since the NSL-KDD-style dataset is flow-based (not raw packet-level time
series), "sequences" here means short sliding windows of consecutive rows
in the preprocessed dataset - a standard, defensible simplification for
this kind of tabular IDS data.

Run on your own laptop (no EC2 needed).

Requirements:
    pip install tensorflow numpy pandas scikit-learn matplotlib joblib
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ---- 1. Load your existing preprocessed data ----
X = pd.read_csv("X_processed.csv")
y_raw = pd.read_csv("y_processed.csv").iloc[:, 0]

# Apply the SAME scaler used during original preprocessing - X_processed.csv
# turned out to contain raw (unscaled) values, and features like src_bytes/
# dst_bytes range into the billions, which would otherwise dominate the
# autoencoder's error calculation and drown out every other feature.
scaler = joblib.load("scaler.pkl")
X = pd.DataFrame(scaler.transform(X), columns=X.columns)

# ---- 2. Build sliding-window sequences ----
WINDOW_SIZE = 10  # each "sequence" = 10 consecutive flows
N_FEATURES = X.shape[1]

def make_sequences(features_df, labels_series, window_size):
    """Turns a flat table into overlapping windows of consecutive rows."""
    feats = features_df.values
    sequences = []
    seq_labels = []  # label = the label of the LAST flow in the window
    for i in range(len(feats) - window_size + 1):
        sequences.append(feats[i:i + window_size])
        seq_labels.append(labels_series.iloc[i + window_size - 1])
    return np.array(sequences), np.array(seq_labels)

X_seq, y_seq = make_sequences(X, y_raw, WINDOW_SIZE)
print(f"Built {len(X_seq)} sequences of shape {X_seq.shape[1:]}")

# ---- 3. Split: train the autoencoder ONLY on Normal sequences ----
normal_mask = (y_seq == "Normal")
X_train_normal = X_seq[normal_mask]
print(f"Training on {len(X_train_normal)} Normal-only sequences")

# Hold out a small validation slice of normal sequences too
split_point = int(len(X_train_normal) * 0.9)
X_train = X_train_normal[:split_point]
X_val = X_train_normal[split_point:]

# ---- 4. Build the LSTM-autoencoder ----
latent_dim = 16

inputs = keras.Input(shape=(WINDOW_SIZE, N_FEATURES))
encoded = layers.LSTM(32, activation="tanh", return_sequences=True)(inputs)
encoded = layers.LSTM(latent_dim, activation="tanh")(encoded)

repeated = layers.RepeatVector(WINDOW_SIZE)(encoded)
decoded = layers.LSTM(32, activation="tanh", return_sequences=True)(repeated)
decoded = layers.TimeDistributed(layers.Dense(N_FEATURES))(decoded)

autoencoder = keras.Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# ---- 5. Train ----
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=20,
    batch_size=64,
    shuffle=True,
    verbose=1,
)

# ---- 6. Compute reconstruction error for ALL sequences (normal + attack) ----
reconstructions = autoencoder.predict(X_seq, batch_size=128, verbose=0)
mse_per_sequence = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))

# ---- 7. Save the model and the reconstruction-error results ----
autoencoder.save("lstm_autoencoder.h5")
joblib.dump({"window_size": WINDOW_SIZE, "n_features": N_FEATURES}, "lstm_ae_config.pkl")

results_df = pd.DataFrame({
    "reconstruction_error": mse_per_sequence,
    "label": y_seq,
})
results_df.to_csv("lstm_reconstruction_errors.csv", index=False)

# ---- 8. Plot the distribution: benign vs attack reconstruction error ----
plt.figure(figsize=(9, 5))
for label in results_df["label"].unique():
    subset = results_df[results_df["label"] == label]["reconstruction_error"]
    plt.hist(subset, bins=50, alpha=0.5, label=label, density=True)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density")
plt.title("LSTM-Autoencoder Reconstruction Error: Normal vs Attack Classes")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_reconstruction_error_distribution.png", dpi=150)
print("Saved: lstm_autoencoder.h5, lstm_reconstruction_errors.csv, "
      "lstm_reconstruction_error_distribution.png")

print("\nMean reconstruction error by class:")
print(results_df.groupby("label")["reconstruction_error"].mean().sort_values(ascending=False))
