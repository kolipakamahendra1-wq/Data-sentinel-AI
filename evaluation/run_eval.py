"""Evaluate detection + investigation on known failure scenarios and clean controls.

    python -m evaluation.run_eval            # rule-based reports (fast, used in CI)
    python -m evaluation.run_eval --live     # Ollama reports (OLLAMA_MODEL, default qwen2.5:3b)
"""
import argparse
import json
import time
from pathlib import Path

from agents.investigator import investigate
from data.generator import DATASETS, FAILURES, KEY_FEATURE, generate, inject
from ml.train import train
from monitoring.checks import run_checks

CONTROL_SEEDS = [1, 2, 3]


def evaluate(live: bool, datasets=DATASETS, failures=FAILURES) -> dict:
    cases = []
    for name in datasets:
        ref = generate(name, 2000, 0)
        model, baseline = train(ref)
        scenarios = [(None, s) for s in CONTROL_SEEDS] + [(kind, 1) for kind in failures]
        for kind, seed in scenarios:
            cur = generate(name, 2000, seed)
            if kind:
                cur = inject(cur, kind)
            check = run_checks(ref, cur, model, baseline)
            case = {"dataset": name, "injected": kind, "seed": seed, "alerted": check["alert"]}
            if check["alert"]:
                start = time.perf_counter()
                report = investigate(check, {}, use_llm=live)
                case.update(latency_s=round(time.perf_counter() - start, 3), predicted=report.root_cause,
                            source=report.source,
                            mentions_feature=KEY_FEATURE[name] in " ".join([report.alert, *report.evidence]))
            cases.append(case)
            print(f"{name:7} {kind or 'clean':19} alerted={case['alerted']!s:5} predicted={case.get('predicted', '-')}")

    failures_ = [c for c in cases if c["injected"]]
    controls = [c for c in cases if not c["injected"]]
    reported = [c for c in failures_ if c["alerted"]]
    # leakage adds a new column and imbalance affects the label, so the key feature is not expected there
    explainable = [c for c in reported if c["injected"] not in ("data_leakage", "label_imbalance")]
    latencies = [c["latency_s"] for c in cases if "latency_s" in c]
    return {
        "mode": "live" if live else "rules",
        "n_failure_scenarios": len(failures_),
        "n_controls": len(controls),
        "detection_accuracy": round(sum(c["alerted"] for c in failures_) / len(failures_), 3),
        "false_positive_rate": round(sum(c["alerted"] for c in controls) / len(controls), 3),
        "root_cause_accuracy": round(sum(c.get("predicted") == c["injected"] for c in failures_) / len(failures_), 3),
        "explanation_mentions_feature": round(sum(c["mentions_feature"] for c in explainable) / max(len(explainable), 1), 3),
        "llm_report_rate": round(sum(c.get("source") == "llm" for c in reported) / max(len(reported), 1), 3),
        "mean_alert_to_report_s": round(sum(latencies) / max(len(latencies), 1), 3),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use Ollama for investigation reports")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    result = evaluate(args.live, args.datasets)
    out = Path(__file__).parent / f"report{'_live' if args.live else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    print(f"wrote {out}")
