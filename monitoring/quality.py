"""Data-quality and schema checks: compare a current batch against the reference batch."""
import re

import pandas as pd
from pydantic import BaseModel

from profiling.profiler import profile

PII_NAMES = re.compile(r"email|phone|ssn|name", re.I)
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class Finding(BaseModel):
    type: str  # missing_values | duplicates | invalid_range | schema_change | cardinality_change | data_leakage
    feature: str | None
    severity: str  # medium | high
    before: float | str | None = None
    after: float | str | None = None
    message: str


def mask_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Replace columns that look like PII with a constant mask."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        looks_like_email = not pd.api.types.is_numeric_dtype(s) and s.dropna().astype(str).head(100).str.fullmatch(EMAIL).mean() > 0.5
        if PII_NAMES.search(col) or looks_like_email:
            df[col] = "***"
    return df


def _label_corr(df, col, label):
    if label not in df or not pd.api.types.is_numeric_dtype(df[col]):
        return 0.0
    c = df[[col, label]].dropna().corr().iloc[0, 1]
    return 0.0 if pd.isna(c) else abs(float(c))


def check_quality(ref: pd.DataFrame, cur: pd.DataFrame, label: str = "label") -> list[Finding]:
    rp, cp = profile(ref), profile(cur)
    rc, cc = rp["columns"], cp["columns"]
    findings = []

    for col in cc.keys() - rc.keys():
        corr = _label_corr(cur, col, label)
        if corr > 0.95:
            findings.append(Finding(type="data_leakage", feature=col, severity="high", after=round(corr, 3),
                                    message=f"New column `{col}` is almost perfectly correlated with the label (|r|={corr:.2f})"))
        else:
            findings.append(Finding(type="schema_change", feature=col, severity="high", after=cc[col]["dtype"],
                                    message=f"Unexpected new column `{col}` appeared"))
    for col in rc.keys() - cc.keys():
        findings.append(Finding(type="schema_change", feature=col, severity="high", before=rc[col]["dtype"],
                                message=f"Expected column `{col}` is missing from the batch"))

    for col in sorted(rc.keys() & cc.keys()):
        r, c = rc[col], cc[col]
        if r["dtype"] != c["dtype"]:
            findings.append(Finding(type="schema_change", feature=col, severity="high", before=r["dtype"],
                                    after=c["dtype"], message=f"Column `{col}` changed type from {r['dtype']} to {c['dtype']}"))
            continue
        if c["missing_frac"] - r["missing_frac"] > 0.05:
            findings.append(Finding(type="missing_values", feature=col,
                                    severity="high" if c["missing_frac"] > 0.2 else "medium",
                                    before=round(r["missing_frac"], 4), after=round(c["missing_frac"], 4),
                                    message=f"Feature `{col}` missingness increased from {r['missing_frac']:.1%} to {c['missing_frac']:.1%}"))
        if r["dtype"] == "numeric" and "min" in r and col != label:
            span = r["max"] - r["min"]
            vals = cur[col].dropna()
            out = float(((vals < r["min"] - span) | (vals > r["max"] + span)).mean()) if len(vals) else 0.0
            if out > 0.01:
                findings.append(Finding(type="invalid_range", feature=col, severity="medium", before=0.0, after=round(out, 4),
                                        message=f"{out:.1%} of `{col}` values are far outside the reference range [{r['min']:.4g}, {r['max']:.4g}]"))
        if r["dtype"] == "categorical" and r["unique"] < 0.5 * rp["n_rows"]:  # skip identifier-like columns
            ratio = c["unique"] / max(r["unique"], 1)
            if abs(c["unique"] - r["unique"]) >= 2 and (ratio > 1.5 or ratio < 0.5):
                findings.append(Finding(type="cardinality_change", feature=col, severity="medium", before=r["unique"],
                                        after=c["unique"], message=f"Feature `{col}` cardinality changed from {r['unique']} to {c['unique']}"))
        if col != label and _label_corr(cur, col, label) > 0.95 and _label_corr(ref, col, label) < 0.8:
            findings.append(Finding(type="data_leakage", feature=col, severity="high",
                                    message=f"Feature `{col}` became almost perfectly correlated with the label"))

    if cp["duplicate_frac"] - rp["duplicate_frac"] > 0.05:
        findings.append(Finding(type="duplicates", feature=None, severity="medium", before=round(rp["duplicate_frac"], 4),
                                after=round(cp["duplicate_frac"], 4),
                                message=f"Duplicate rows rose from {rp['duplicate_frac']:.1%} to {cp['duplicate_frac']:.1%}"))
    return findings
