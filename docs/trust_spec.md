\# Trust-score specification (v1, Phase 2)



\## 1. Evidence score

p(x) = calibrated probability that flow x is an attack = 1 - p\_hat(benign | x).

It comes from the stacked ensemble: RF class probabilities, Isolation Forest

score, and LSTM-AE error (only if it passes the keep rule in section 6). The

meta-learner is trained on out-of-fold outputs and then calibrated (isotonic

or Platt) on held-out calibration data.

T(x) = 1 - p(x). There is no second weighted sum. Drift D(t) is NOT an input.



\## 2. Conformal evidence classes

Per-class (Mondrian) split conformal, alpha in {0.01, 0.05, 0.10}.

\- confident-benign: prediction set = {benign}

\- confident-attack: benign excluded, only attack classes remain

\- ambiguous: everything else

Coverage is measured on held-out data with confidence intervals, never assumed.



\## 3. Entity trust

Te(t) = lambda \* Te(t-1) + (1 - lambda) \* T(x\_t)

lambda = lambda\_down if T(x\_t) < Te(t-1), else lambda\_up.

Fast distrust, slow recovery. Initial trust T0 is tunable. Entity = source IP.



\## 4. Zones

\- Allow: Te >= tau\_h

\- Alert + rate-limit: tau\_l <= Te < tau\_h, or ambiguous evidence

\- Block: Te < tau\_l AND at least k\_eff confident-attack flows in the last W flows

\- Re-admission: after cool-down t\_c, only when Te >= tau\_l + delta



\## 5. Drift guard (outside the trust score)

D(t) = max(D\_feat, D\_amb). Effects only:

\- alpha\_eff = alpha \* (1 - kappa \* D)  (wider ambiguous zone)

\- k\_eff = ceil(k \* (1 + D))  (tighter block gate)

\- retraining alert when D stays high

D(t) never lowers T(x) or Te.



\## 6. LSTM-AE keep rule (set before results)

Keep it only if adding it to RF + IF raises macro-F1 with a bootstrap 95% CI

that excludes zero, or raises rare-class recall by at least 3 points with a CI

that excludes zero. Otherwise drop it and report the result.



\## 7. Primary metrics

macro-F1, false-block rate, time-to-block, p95 detection-to-block latency.

Everything else is secondary. Holm correction only within this family.



\## 8. Baselines

\- B1: plain Random Forest, hard labels

\- B2: static per-flow threshold, block when T(x) < 0.5, no entity state

\- B3: Phase 1 formula 1 - \[0.55\*y\_hat + 0.40\*sigmoid(s) + 0.05\*D], uncalibrated



\## 9. Changes from Phase 1

Drift removed from T(x); one fusion layer; Phase 1 formula kept as B3.

Reason: the sigmoid of the Isolation Forest score only varies between 0.58 and

0.67 on the Phase 1 data, so its weight acts like a threshold shift.

