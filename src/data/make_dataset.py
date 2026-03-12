# src/data/make_dataset.py

import numpy as np
import pandas as pd

from src.features.subfolder.schema import FEATURES, TARGET

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def generate_synthetic_fraud(
    n_rows: int,
    seed: int = 42,
    drift_mode: str = "none",  # none | mild | severe
) -> pd.DataFrame:
    """
    Deterministic synthetic dataset generator for fraud-like transactions.

    Notes:
    - No PII.
    - Label is probabilistic and derived from an interpretable risk function.
    - drift_mode changes feature distributions to simulate data drift.
    """
    rng = np.random.default_rng(seed)

    # Base feature distributions
    amount = rng.lognormal(mean=3.4, sigma=1.0, size=n_rows)            # long-tailed
    hour = rng.integers(0, 24, size=n_rows)
    country_risk = rng.integers(1, 6, size=n_rows)                      # 1..5
    is_cross_border = rng.binomial(1, 0.35, size=n_rows)
    customer_tenure_days = rng.integers(1, 3650, size=n_rows)
    txns_7d = rng.poisson(lam=3.0, size=n_rows)
    chargebacks_365d = rng.poisson(lam=0.15, size=n_rows)

    # Drift scenarios: shift some distributions (fraud patterns evolve)
    if drift_mode == "mild":
        amount = amount * rng.normal(1.05, 0.05, size=n_rows)
        is_cross_border = rng.binomial(1, 0.42, size=n_rows)
    elif drift_mode == "severe":
        amount = amount * rng.normal(1.25, 0.10, size=n_rows)
        is_cross_border = rng.binomial(1, 0.60, size=n_rows)
        country_risk = rng.integers(2, 6, size=n_rows)                  # fewer low-risk

    # Interpretable risk function (log-odds style)
    night = ((hour <= 5) | (hour >= 23)).astype(int)

    z = (
        0.0025 * amount.clip(0, 25000)
        + 0.45 * is_cross_border
        + 0.35 * (country_risk - 1)
        + 0.18 * np.log1p(txns_7d)
        + 0.55 * np.log1p(chargebacks_365d)
        - 0.00025 * customer_tenure_days
        + 0.10 * night
        - 2.6
    )

    p = _sigmoid(z)
    is_fraud = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "amount": amount.astype(float),
            "hour": hour.astype(int),
            "country_risk": country_risk.astype(int),
            "is_cross_border": is_cross_border.astype(int),
            "customer_tenure_days": customer_tenure_days.astype(int),
            "txns_7d": txns_7d.astype(int),
            "chargebacks_365d": chargebacks_365d.astype(int),
            TARGET: is_fraud.astype(int),
        }
    )

    # Ensure column order (nice for reproducibility)
    df = df[FEATURES + [TARGET]]
    return df


