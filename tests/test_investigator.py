import pytest

from agents import investigator
from agents.investigator import Report, investigate, rule_report
from data.generator import FAILURES, generate, inject
from ml.train import train
from monitoring.checks import run_checks

REF = generate("churn", 2000, 0)
MODEL, BASELINE = train(REF)


def check(kind=None):
    cur = generate("churn", 2000, 1)
    return run_checks(REF, inject(cur, kind) if kind else cur, MODEL, BASELINE)


def test_clean_batch_raises_no_alert():
    result = check()
    assert not result["alert"]
    assert result["health_score"] == 100


@pytest.mark.parametrize("kind", FAILURES)
def test_rules_identify_root_cause(kind):
    result = check(kind)
    assert result["alert"]
    assert rule_report(result, {}).root_cause == kind


def test_rule_report_cites_recent_deployment():
    ctx = {"events": [{"type": "deployment", "message": "feature-pipeline v2.3 deployed", "created_at": "2026-10-09T10:00"}]}
    report = rule_report(check("missing_values"), ctx)
    assert any("v2.3" in e for e in report.evidence)
    assert "income" in report.alert


def test_falls_back_to_rules_when_ollama_unreachable(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    report = investigate(check("outliers"), {})
    assert report.source == "rules"
    assert report.root_cause == "outliers"


def test_falls_back_on_invalid_llm_output(monkeypatch):
    monkeypatch.setattr(investigator, "_ollama_chat", lambda messages: '{"alert": "x"}')
    assert investigate(check("schema_change"), {}).source == "rules"


def test_uses_valid_llm_output(monkeypatch):
    body = Report(alert="a", root_cause="schema_change", likely_cause="c", evidence=["e"],
                  impact="i", recommended_action="r", confidence=0.7).model_dump_json(exclude={"source"})
    monkeypatch.setattr(investigator, "_ollama_chat", lambda messages: body)
    report = investigate(check("schema_change"), {})
    assert report.source == "llm" and report.likely_cause == "c"


def test_llm_percentage_confidence_is_normalised(monkeypatch):
    body = ('{"alert": "a", "root_cause": "outliers", "likely_cause": "c", "evidence": ["e"], '
            '"impact": "i", "recommended_action": "r", "confidence": 95}')
    monkeypatch.setattr(investigator, "_ollama_chat", lambda messages: body)
    report = investigate(check("outliers"), {})
    assert report.source == "llm" and report.confidence == 0.95


def test_new_leaking_column_lowers_health():
    result = check("data_leakage")
    assert result["feature_health"]["approval_score"] == "bad"
    assert result["health_score"] < 100
