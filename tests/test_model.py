from data.generator import generate, inject
from ml.model_metrics import compare, evaluate
from ml.train import train


def test_baseline_metrics_reasonable():
    model, baseline = train(generate("loan", 3000, 0))
    assert baseline["roc_auc"] > 0.7
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "ece", "false_positives", "false_negatives"]:
        assert k in baseline


def test_clean_batch_not_degraded():
    model, baseline = train(generate("loan", 3000, 0))
    cur = evaluate(model, generate("loan", 2000, 1))
    assert not compare(baseline, cur)["degraded"]


def test_handles_missing_column():
    model, _ = train(generate("demand", 2000, 0))
    cur = evaluate(model, inject(generate("demand", 2000, 1), "schema_change"))
    assert 0 <= cur["roc_auc"] <= 1


def test_shift_degrades_model():
    model, baseline = train(generate("loan", 3000, 0))
    bad = generate("loan", 2000, 1)
    bad["credit_score"] = bad["credit_score"] - 150
    result = compare(baseline, evaluate(model, bad))
    assert result["degraded"]
    assert result["deltas"]["accuracy"] < 0
