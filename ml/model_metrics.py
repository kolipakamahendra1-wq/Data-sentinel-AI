"""Model-performance metrics and degradation check."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

TRACKED = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _ece(y, p, bins: int = 10) -> float:
    """Expected calibration error."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def evaluate(model, df: pd.DataFrame, label: str = "label") -> dict:
    X = df.reindex(columns=model.feature_names_in_)  # missing columns become NaN and get imputed
    y = df[label].to_numpy()
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else 0.5,
        "ece": _ece(y, p),
        "mean_confidence": float(np.maximum(p, 1 - p).mean()),
        "false_positives": int(((pred == 1) & (y == 0)).sum()),
        "false_negatives": int(((pred == 0) & (y == 1)).sum()),
    }


def compare(baseline: dict, current: dict) -> dict:
    deltas = {k: round(current[k] - baseline[k], 4) for k in TRACKED + ["ece", "mean_confidence"]}
    degraded = deltas["roc_auc"] < -0.1 or deltas["accuracy"] < -0.05 or deltas["f1"] < -0.1
    return {"degraded": degraded, "deltas": deltas, "current": current}
