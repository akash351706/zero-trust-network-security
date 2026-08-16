import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import joblib
import matplotlib.pyplot as plt

X = pd.read_csv("X_processed.csv")
y = pd.read_csv("y_processed.csv")["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train[y_train == "Normal"])
print("Isolation Forest done.")

print("Running GridSearchCV for Random Forest (this takes a few minutes)...")
params = {"n_estimators": [100, 200], "max_depth": [10, 20]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, scoring="f1_macro")
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print("Best parameters:", grid.best_params_)

y_pred = best_rf.predict(X_test)
print("\n=== Per-Class Evaluation Report ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_rf, X, y, cv=cv, scoring="f1_macro")
print("5-fold F1-macro scores:", scores)
print(f"Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")

print("\nComputing mutual information feature ranking...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print(mi_series.head(15))

plt.figure(figsize=(10,6))
mi_series.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 Features by Mutual Information")
plt.xlabel("Mutual Information Score")
plt.tight_layout()
plt.savefig("mi_feature_ranking.png")
print("Saved: mi_feature_ranking.png")

joblib.dump(best_rf, "random_forest_model.pkl")
joblib.dump(iso_forest, "isolation_forest_model.pkl")
print("\nSaved: random_forest_model.pkl, isolation_forest_model.pkl")