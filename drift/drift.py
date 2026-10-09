"""Distribution drift: PSI + KS per feature, and label drift."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4


def _dist_psi(p, q):
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(ref, cur, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the reference."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    return _dist_psi(p, q)


def categorical_psi(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()), key=str)
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    q = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _dist_psi(p, q)


def check_drift(ref: pd.DataFrame, cur: pd.DataFrame, label: str = "label") -> dict:
    features = []
    for col in [c for c in ref.columns if c in cur.columns and c != label]:
        r, c = ref[col].dropna(), cur[col].dropna()
        if len(r) == 0 or len(c) == 0:
            continue
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(c):
            score = psi(r, c)
            ks = ks_2samp(r, c)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        else:
            if r.nunique() > 0.5 * len(r):  # identifier-like column
                continue
            score, ks_stat, ks_p = categorical_psi(r, c), None, None
        drifted = score >= 0.2 or (ks_p is not None and ks_p < 0.01 and score >= 0.1)
        features.append({"feature": col, "psi": round(score, 4), "ks_stat": ks_stat and round(ks_stat, 4),
                         "ks_p": ks_p, "drifted": drifted})
    label_result = {"drifted": False, "ref_rate": None, "cur_rate": None}
    if label in ref and label in cur:
        rr, cr = float(ref[label].mean()), float(cur[label].mean())
        label_result = {"ref_rate": round(rr, 4), "cur_rate": round(cr, 4), "drifted": abs(cr - rr) > 0.1}
    return {"features": features, "label": label_result,
            "max_psi": max((f["psi"] for f in features), default=0.0)}
