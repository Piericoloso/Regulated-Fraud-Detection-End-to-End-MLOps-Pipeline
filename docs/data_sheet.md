# Data Sheet — Synthetic AML-like Transactions

## Motivation
Provide a license-safe, reproducible dataset for regulated MLOps demos.

## Composition
Rows: configurable (default 25k). Label is_suspicious (0/1).

## Collection process
Generated via scripted distributions + transparent scoring function + Bernoulli sampling.

## Preprocessing
Standard scaling for Logistic Regression.

## Retention & privacy
No PII. No retention constraints. Mirrors governance structure used with real data.
