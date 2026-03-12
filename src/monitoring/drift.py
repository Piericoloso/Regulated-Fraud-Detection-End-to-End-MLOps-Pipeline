import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    # numeric PSI with quantile bins from expected
    expected = expected.astype(float)
    actual = actual.astype(float)

    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = expected.quantile(quantiles).values
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, 1)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, 1)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
