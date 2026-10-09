"""AI investigation agent: turns a monitoring result + context into an incident report.

Uses a local Ollama model with JSON-schema constrained output; falls back to deterministic rules
when the model is unreachable or returns something that does not validate.
"""
import json
import os
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

RootCause = Literal["missing_values", "distribution_shift", "label_imbalance", "schema_change",
                    "outliers", "data_leakage", "unknown"]


class Report(BaseModel):
    alert: str
    root_cause: RootCause
    likely_cause: str
    evidence: list[str]
    impact: str
    recommended_action: str
    confidence: float = Field(ge=0, le=1)
    source: Literal["llm", "rules"] = "rules"

    @field_validator("confidence", mode="before")
    @classmethod
    def _percent_to_fraction(cls, v):
        # small models often answer 95 instead of 0.95
        return v / 100 if isinstance(v, (int, float)) and 1 < v <= 100 else v


ACTIONS = {
    "missing_values": ("Upstream pipeline or schema change dropping values",
                       "Roll back the upstream change or fix the feature transformation; impute only as a stopgap."),
    "distribution_shift": ("Real-world population shift or unit/scaling change upstream",
                           "Confirm units/scaling upstream; if the shift is genuine, retrain on recent data."),
    "label_imbalance": ("Sampling or labelling change altered the class balance",
                        "Check sampling filters and labelling jobs; recalibrate the decision threshold or retrain."),
    "schema_change": ("Upstream schema change renamed or removed a column",
                      "Roll back the schema change or update the feature mapping, then add a schema contract test."),
    "outliers": ("Corrupted or mis-scaled values entering the pipeline",
                 "Add range validation at ingestion and quarantine the bad records."),
    "data_leakage": ("A feature that encodes the target was added to the input",
                     "Remove the leaking column from the feature set immediately and audit feature lineage."),
    "unknown": ("Model degradation without a clear data-quality signal",
                "Investigate recent model/deployment changes and compare against the baseline model."),
}


def _root_cause(check: dict) -> tuple[str, str | None]:
    findings = check["findings"]
    for ftype, cause in [("data_leakage", "data_leakage"), ("schema_change", "schema_change"),
                         ("missing_values", "missing_values"), ("invalid_range", "outliers")]:
        hit = next((f for f in findings if f["type"] == ftype), None)
        if hit:
            return cause, hit["feature"]
    if check["drift"]["label"]["drifted"]:
        return "label_imbalance", "label"
    drifted = sorted((f for f in check["drift"]["features"] if f["drifted"]), key=lambda f: -f["psi"])
    if drifted:
        return "distribution_shift", drifted[0]["feature"]
    return "unknown", None


def _impact(check: dict) -> str:
    m = check.get("model")
    if not m:
        return "Model impact unknown (no labelled model available for this dataset)."
    d = m["deltas"]
    verdict = "degraded" if m["degraded"] else "not significantly degraded"
    return (f"Model performance {verdict}: ROC-AUC {d['roc_auc']:+.3f}, accuracy {d['accuracy']:+.3f}, "
            f"F1 {d['f1']:+.3f}, mean confidence {d['mean_confidence']:+.3f}.")


def rule_report(check: dict, context: dict) -> Report:
    cause, feature = _root_cause(check)
    evidence = [f["message"] for f in check["findings"][:4]]
    if check["drift"]["label"]["drifted"]:
        lbl = check["drift"]["label"]
        evidence.append(f"Label positive rate moved from {lbl['ref_rate']:.1%} to {lbl['cur_rate']:.1%}")
    drifted = [f["feature"] for f in check["drift"]["features"] if f["drifted"]]
    if drifted:
        evidence.append(f"Drifted features: {', '.join(drifted)}")
    if feature and len(drifted) <= 1:
        others = [f for f in check["feature_health"] if f not in drifted and f != feature]
        evidence.append(f"Other features unaffected ({len(others)} checked)")
    for e in context.get("events", [])[:2]:
        evidence.append(f"Recent {e['type']} event at {e['created_at']}: {e['message']}")
    likely, action = ACTIONS[cause]
    return Report(alert=check["title"] or "No anomaly detected", root_cause=cause, likely_cause=likely,
                  evidence=evidence, impact=_impact(check), recommended_action=action,
                  confidence=0.8 if cause != "unknown" else 0.3)


def _ollama_chat(messages: list[dict]) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    schema = Report.model_json_schema()
    schema["properties"].pop("source")
    resp = httpx.post(f"{url}/api/chat", timeout=float(os.getenv("OLLAMA_TIMEOUT", "120")), json={
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), "messages": messages, "stream": False,
        "format": schema, "options": {"temperature": 0},
    })
    resp.raise_for_status()
    return resp.json()["message"]["content"]


SYSTEM = """You are DataSentinel, an ML reliability investigator. You receive monitoring evidence about a
dataset batch (data-quality findings, drift statistics, model metric deltas, pipeline/deployment events,
previous incidents). Explain the alert for an ML engineer. Use only the evidence given and cite concrete numbers.
root_cause must be one of: missing_values, distribution_shift, label_imbalance, schema_change, outliers,
data_leakage, unknown. detector_hypothesis is a rule-based guess; agree with it unless the evidence
contradicts it. If pipeline_events are present, cite the most recent one in evidence, since a
failure that starts right after a deployment points to that deployment. confidence is a number between 0 and 1. Respond with JSON only."""


def llm_report(check: dict, context: dict) -> Report:
    hypothesis, feature = _root_cause(check)
    evidence = {
        "alert": check["title"],
        "detector_hypothesis": {"root_cause": hypothesis, "feature": feature},
        "findings": [f["message"] for f in check["findings"]],
        "drifted_features": [f for f in check["drift"]["features"] if f["drifted"]],
        "label": check["drift"]["label"],
        "model": check["model"] and {"degraded": check["model"]["degraded"], "deltas": check["model"]["deltas"]},
        "pipeline_events": context.get("events", []),
        "previous_incidents": context.get("previous_incidents", []),
    }
    content = _ollama_chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": json.dumps(evidence, default=str)}])
    return Report.model_validate_json(content).model_copy(update={"source": "llm"})


def investigate(check: dict, context: dict, use_llm: bool = True) -> Report:
    if use_llm:
        try:
            return llm_report(check, context)
        except (httpx.HTTPError, ValidationError, KeyError, ValueError):
            pass
    return rule_report(check, context)
