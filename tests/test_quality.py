import pandas as pd
import pytest

from data.generator import DATASETS, generate, inject
from monitoring.quality import check_quality, mask_pii


def types(findings):
    return {f.type for f in findings}


@pytest.mark.parametrize("name", DATASETS)
def test_clean_batch_has_no_findings(name):
    assert check_quality(generate(name, 2000, 0), generate(name, 2000, 1)) == []


@pytest.mark.parametrize("kind,expected", [
    ("missing_values", "missing_values"),
    ("schema_change", "schema_change"),
    ("outliers", "invalid_range"),
    ("data_leakage", "data_leakage"),
])
def test_injected_failure_detected(kind, expected):
    ref = generate("churn", 2000, 0)
    cur = inject(generate("churn", 2000, 1), kind)
    assert expected in types(check_quality(ref, cur))


def test_missing_finding_names_feature_and_rates():
    ref = generate("loan", 2000, 0)
    f = [f for f in check_quality(ref, inject(generate("loan", 2000, 1), "missing_values"))
         if f.type == "missing_values"][0]
    assert f.feature == "income"
    assert f.before < 0.01 and f.after > 0.2
    assert "income" in f.message


def test_duplicates_detected():
    ref = generate("churn", 1000, 0)
    cur = generate("churn", 1000, 1)
    cur = pd.concat([cur, cur.iloc[:300]], ignore_index=True)
    assert "duplicates" in types(check_quality(ref, cur))


def test_cardinality_change_detected():
    ref = generate("churn", 1000, 0)
    cur = generate("churn", 1000, 1)
    cur["contract"] = [f"c{i % 40}" for i in range(len(cur))]
    assert "cardinality_change" in types(check_quality(ref, cur))


def test_mask_pii_hides_email_column():
    df = mask_pii(generate("churn", 50, 0))
    assert set(df["email"]) == {"***"}


def test_mask_pii_detects_emails_by_value():
    df = generate("churn", 50, 0).rename(columns={"email": "contact"})
    assert set(mask_pii(df)["contact"]) == {"***"}
    assert mask_pii(df)["contract"].equals(df["contract"])
