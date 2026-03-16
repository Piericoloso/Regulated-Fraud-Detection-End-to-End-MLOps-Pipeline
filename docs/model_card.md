# Model Card — fincrime_suspicious_txn

## Intended use
Score transactions for suspected suspicious activity triage (analyst review). Not for automated enforcement.

## Out of scope
Do not use for identity proofing, sanctions decisions, or adverse action without human review.

## Data
Synthetic dataset generated deterministically (seeded). No real customer data. Features mimic AML-style risk signals.

## Features
- amount, hour, country_risk, is_cross_border, customer_tenure_days, txns_7d, chargebacks_365d

## Performance
Tracked in MLflow:
- ROC-AUC
- PR-AUC
- Brier (calibration proxy)

## Threshold policy
Default 0.50 for demo. Production requires business calibration + model risk approval and periodic review.

## Explainability
Global: coefficients / permutation importance. Local: (optional) SHAP in a later milestone.

## Monitoring
- Request logging + latency
- Drift: PSI report generated daily from baseline vs recent traffic

## Rollback
Promote previous Production model version in MLflow Registry; redeploy API with stage=Production unchanged.
