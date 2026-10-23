import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X = pd.read_csv("X_processed.csv")
y = pd.read_csv("y_processed.csv")["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = joblib.load("random_forest_model.pkl")
iso = joblib.load("isolation_forest_model.pkl")

print("RF macro-F1 on TRAIN rows:", round(f1_score(ytr, rf.predict(Xtr), average="macro"), 4))
print("RF macro-F1 on TEST rows :", round(f1_score(yte, rf.predict(Xte), average="macro"), 4))

def trust_f1(Xs, ys):
    p = 1 - rf.predict_proba(Xs)[:, list(rf.classes_).index("Normal")]
    raw = -iso.score_samples(Xs)
    s = 1 / (1 + np.exp(-raw))
    T = 1 - (0.55 * p + 0.40 * s)
    return f1_score((ys != "Normal").astype(int), (T < 0.55).astype(int))

print("Trust-score F1 on ALL rows (as in repo):", round(trust_f1(X, y), 4))
print("Trust-score F1 on TEST rows only       :", round(trust_f1(Xte, yte), 4))