AI-Powered Zero Trust Network Security System on Cloud
A Trust-Score-Based Adaptive Zero Trust Framework for Real-Time Intrusion Detection
Final-year ME (Communication Systems) project | Akash B | Reg. No: 2508110001 | Rajalakshmi Engineering College, Dept. of ECE | Guide: Dr. A. Asha
Status: Phase 1 (Detection Engine) — Complete. IEEE conference paper based on this work has been accepted for publication.
---
Project Overview
This project builds a network intrusion detection system that goes beyond a single classifier by combining three models — a Random Forest, an Isolation Forest, and an LSTM-Autoencoder — into a stacked ensemble whose output feeds a continuous trust score for every network flow. The system is validated on the NSL-KDD benchmark, backed by real live packet capture on AWS EC2, statistically validated with 5-fold cross-validation and significance testing, and made interpretable using SHAP.
What's in this repository
1. Data Preprocessing
`preprocess.py` — cleans raw NSL-KDD data, encodes categorical fields, maps attacks into 5 classes (Normal, DoS, PortScan, BruteForce, Data Exfiltration), and scales features
`X_processed.csv`, `y_processed.csv` — the resulting processed dataset
`scaler.pkl` — the fitted StandardScaler, reused across all later models
2. Classical ML Models
`train.py` — trains and tunes Isolation Forest and Random Forest (via GridSearchCV), with 5-fold cross-validation and mutual-information feature ranking
`isolation_forest_model.pkl`, `random_forest_model.pkl` — trained model files
`mi_feature_ranking.png` — mutual information feature importance chart
`results_summary.txt` — per-class precision/recall/F1 and CV results
3. Live Packet Capture (AWS EC2)
`capture.py` — Scapy-based live packet sniffer, run on an AWS EC2 Ubuntu instance inside a custom VPC
`captured_traffic.csv` — 58,571 real packets captured live from the running instance (timestamp, source/destination IP, protocol, ports, packet size)
4. Literature Review & Novelty
`Literature_Gap_Table.docx` — comparison of 10 papers across cloud IDS, ensemble IDS, deep-learning IDS, and drift-aware IDS categories, with an explicit novelty statement for this project
5. Formal System Model & Trust Score
`System_Model_TrustScore.docx` — formal hypothesis-testing framing of detection, the trust-score function T(x), a queueing-based latency model, and the feature-selection justification
`trust_score_weight_search.py` — grid search that found the tuned weights: w1=0.55 (Random Forest), w2=0.40 (Isolation Forest), w3=0.05 (drift term placeholder), achieving F1=0.9997 on validation data at threshold T(x) < 0.55
6. LSTM-Autoencoder Temporal Baseline
`train_lstm_autoencoder.py` — trains an LSTM-autoencoder on sliding windows of benign-only flow sequences; reconstruction error is used as a temporal anomaly signal
`lstm_autoencoder.h5` — trained model
`lstm_reconstruction_errors.csv`, `lstm_reconstruction_error_distribution.png` — results showing DataExfiltration, BruteForce, and PortScan sequences reconstruct with visibly higher error than Normal traffic
7. Stacked Ensemble
`train_stacked_ensemble.py` — combines Random Forest, Isolation Forest, and LSTM-Autoencoder outputs via a Logistic Regression meta-learner
`stacked_ensemble_meta_learner.pkl` — final trained ensemble model, achieving F1 = 0.9997
8. Statistical Validation
`significance_testing.py` — 5-fold cross-validation comparing the ensemble against each base learner individually, with paired t-test and Wilcoxon signed-rank significance testing
`significance_test_fold_results.csv` — per-fold F1 scores for Random Forest alone, Isolation Forest alone, and the full ensemble
`baseline_comparison_table.csv` — comparison against 3 published results on NSL-KDD (Roy et al., De Souza et al., and a CNN+Autoencoder ensemble paper)
9. Explainability
`shap_explainability.py` — SHAP analysis on both the Random Forest and the meta-learner
`shap_random_forest_summary.png` — which raw features drive Random Forest's malicious-flow predictions
`shap_meta_learner_summary.png` — which of the three signals (RF / IF / LSTM-AE) drive the final ensemble decision
10. Publication
IEEE conference paper based on this work — accepted
Key Results
Component	Result
Random Forest + Isolation Forest (5-fold CV)	0.9191 mean F1
Trust-score weight search	w1=0.55, w2=0.40, w3=0.05 — F1=0.9997
LSTM-AE reconstruction error (Normal vs Attack)	Normal/DoS ≈ 0.59, PortScan/BruteForce ≈ 0.75-0.76, DataExfiltration ≈ 1.95
Stacked Ensemble (RF + IF + LSTM-AE)	F1 = 0.9997
Novelty Statement
No reviewed work combines a trust-score-fused ensemble (classical + temporal deep learning) with statistically validated, drift-aware evaluation on standard IDS benchmarks — this project closes that gap.
Honest Limitations
The drift-adjustment term D(x) in the trust-score function is currently a placeholder (held at 0); real drift detection is planned for Phase 2
The Random Forest signal dominates the stacked ensemble's learned weights on this benchmark, reflecting NSL-KDD's known separability rather than a flaw in the ensemble design
Live packet capture demonstrates the pipeline works on real traffic, but the captured sample is not yet integrated into the trained models as additional training data
What's Next (Phase 2)
Enforcement logic (acting on trust-score thresholds), a real drift monitor for D(x), SDN integration for actual flow control, and a live monitoring dashboard — covered in the companion Phase 2 document.
Setup
```bash
pip install scikit-learn joblib numpy pandas matplotlib scapy tensorflow shap scipy
python preprocess.py
python train.py
python trust_score_weight_search.py
python train_lstm_autoencoder.py
python train_stacked_ensemble.py
python significance_testing.py
python shap_explainability.py
```
