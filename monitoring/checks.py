"""One monitoring run: quality + drift + model checks, summarised into health and an alert decision."""
import pandas as pd

from drift.drift import check_drift
from ml.model_metrics import compare, evaluate
from monitoring.quality import check_quality

# Order in which problems are surfaced as the alert headline (most specific first).
PRIORITY = ["data_leakage", "schema_change", "missing_values", "invalid_range", "duplicates", "cardinality_change"]


def run_checks(ref: pd.DataFrame, cur: pd.DataFrame, model=None, baseline: dict | None = None,
               label: str = "label") -> dict:
    findings = sorted(check_quality(ref, cur, label), key=lambda f: PRIORITY.index(f.type))
    drift = check_drift(ref, cur, label)
    model_result = compare(baseline, evaluate(model, cur, label)) if model is not None and label in cur else None

    drifted = [f for f in drift["features"] if f["drifted"]]
    titles = [f.message for f in findings]
    if drift["label"]["drifted"]:
        titles.append(f"Label positive rate shifted from {drift['label']['ref_rate']:.1%} to {drift['label']['cur_rate']:.1%}")
    titles += [f"Feature `{f['feature']}` drifted (PSI {f['psi']:.2f})" for f in drifted]
    if model_result and model_result["degraded"]:
        titles.append(f"Model ROC-AUC changed by {model_result['deltas']['roc_auc']:+.3f}")

    features = [c for c in ref.columns if c != label] + [c for c in cur.columns if c not in ref.columns]
    bad = {f.feature for f in findings if f.severity == "high"} | {f["feature"] for f in drifted}
    warn = {f.feature for f in findings} - bad
    feature_health = {c: "bad" if c in bad else "warn" if c in warn else "ok" for c in features}
    ok = sum(v == "ok" for v in feature_health.values())

    severe = (any(f.severity == "high" for f in findings) or drift["label"]["drifted"]
              or bool(model_result and model_result["degraded"]))
    return {
        "alert": bool(titles),
        "title": titles[0] if titles else None,
        "severity": "high" if severe else "medium" if titles else None,
        "findings": [f.model_dump() for f in findings],
        "drift": drift,
        "model": model_result,
        "feature_health": feature_health,
        "health_score": round(100 * ok / max(len(features), 1)),
        "rows": len(cur),
    }
