from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    pos_rate: float

def evaluate_binary(y_true, y_prob) -> Metrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        pos_rate=float(y_true.mean()),
    )
