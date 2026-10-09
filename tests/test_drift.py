import numpy as np
import pytest

from data.generator import DATASETS, generate, inject
from drift.drift import check_drift, psi


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 1e-6


def test_psi_large_for_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=2000), rng.normal(2, 1, size=2000)) > 0.2


@pytest.mark.parametrize("name", DATASETS)
def test_clean_resample_not_drifted(name):
    result = check_drift(generate(name, 2000, 0), generate(name, 2000, 1))
    assert not any(f["drifted"] for f in result["features"])
    assert not result["label"]["drifted"]


def test_shift_detected_on_key_feature():
    result = check_drift(generate("churn", 2000, 0), inject(generate("churn", 2000, 1), "distribution_shift"))
    drifted = {f["feature"] for f in result["features"] if f["drifted"]}
    assert drifted == {"income"}


def test_label_drift_detected():
    result = check_drift(generate("fraud", 2000, 0), inject(generate("fraud", 2000, 1), "label_imbalance"))
    assert result["label"]["drifted"]
