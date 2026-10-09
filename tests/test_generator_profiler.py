import numpy as np
import pytest

from data.generator import DATASETS, KEY_FEATURE, FAILURES, generate, inject
from profiling.profiler import profile


@pytest.mark.parametrize("name", DATASETS)
def test_generate_has_label_and_key_feature(name):
    df = generate(name, n=500, seed=0)
    assert len(df) == 500
    assert set(df["label"].unique()) <= {0, 1}
    assert 0.02 < df["label"].mean() < 0.6
    assert KEY_FEATURE[name] in df.columns


def test_generate_is_deterministic():
    assert generate("churn", 200, seed=3).equals(generate("churn", 200, seed=3))


def test_inject_missing_values():
    df = inject(generate("loan", 1000, 0), "missing_values")
    assert df["income"].isna().mean() > 0.2


def test_inject_distribution_shift():
    base = generate("churn", 1000, 0)
    shifted = inject(base, "distribution_shift")
    assert shifted["income"].mean() > base["income"].mean() + base["income"].std()


def test_inject_label_imbalance():
    base = generate("fraud", 2000, 0)
    assert inject(base, "label_imbalance")["label"].mean() > base["label"].mean() + 0.2


def test_inject_schema_change_renames_key_feature():
    df = inject(generate("demand", 300, 0), "schema_change")
    assert "price" not in df.columns and "price_v2" in df.columns


def test_inject_outliers():
    base = generate("churn", 1000, 0)
    assert inject(base, "outliers")["income"].max() > base["income"].max() * 3


def test_inject_data_leakage_adds_label_proxy():
    df = inject(generate("loan", 1000, 0), "data_leakage")
    extra = set(df.columns) - set(generate("loan", 10, 0).columns)
    assert len(extra) == 1
    assert abs(np.corrcoef(df[extra.pop()], df["label"])[0, 1]) > 0.95


def test_inject_does_not_mutate_input():
    base = generate("churn", 300, 0)
    copy = base.copy()
    for kind in FAILURES:
        inject(base, kind)
    assert base.equals(copy)


def test_profile_reports_types_and_missingness():
    df = inject(generate("churn", 500, 0), "missing_values")
    p = profile(df)
    assert p["n_rows"] == 500
    assert p["columns"]["income"]["dtype"] == "numeric"
    assert p["columns"]["contract"]["dtype"] == "categorical"
    assert p["columns"]["income"]["missing_frac"] > 0.2


@pytest.mark.parametrize("name", DATASETS)
def test_inject_targets_the_dataset_key_feature(name):
    df = inject(generate(name, 500, 0), "missing_values")
    assert df.columns[df.isna().mean() > 0.2].tolist() == [KEY_FEATURE[name]]
