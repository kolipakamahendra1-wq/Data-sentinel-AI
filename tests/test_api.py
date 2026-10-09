import os
import tempfile

os.environ["DATASENTINEL_STORAGE"] = tempfile.mkdtemp()
os.environ["DATASENTINEL_API_KEYS"] = "key-a:acme,key-b:globex"
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"  # unreachable: forces rule-based reports

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)
A = {"X-API-Key": "key-a"}
B = {"X-API-Key": "key-b"}


def make_dataset(headers=A, generator="loan"):
    r = client.post("/datasets", data={"name": f"{generator}-ref", "generator": generator}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_requires_api_key():
    assert client.get("/alerts").status_code == 401
    assert client.get("/alerts", headers={"X-API-Key": "nope"}).status_code == 401


def test_clean_validation_has_no_alert():
    ds = make_dataset()
    run = client.post("/validation/run", data={"dataset_id": ds, "seed": 1}, headers=A).json()
    assert run["alert_id"] is None
    health = client.get(f"/datasets/{ds}/health", headers=A).json()
    assert health["health_score"] == 100


def test_failure_flow_creates_alert_and_investigation():
    ds = make_dataset()
    client.post("/events", json={"dataset_id": ds, "type": "deployment", "message": "pipeline v2.3"}, headers=A)
    run = client.post("/validation/run", data={"dataset_id": ds, "inject": "missing_values", "seed": 1}, headers=A).json()
    alert_id = run["alert_id"]
    assert alert_id

    alerts = client.get("/alerts", headers=A).json()
    assert any(a["id"] == alert_id and a["status"] == "open" for a in alerts)

    report = client.post("/investigations", json={"alert_id": alert_id}, headers=A).json()
    assert report["root_cause"] == "missing_values"
    assert any("v2.3" in e for e in report["evidence"])

    alert = client.get(f"/alerts/{alert_id}", headers=A).json()
    assert alert["status"] == "investigated" and alert["report"]["root_cause"] == "missing_values"


def test_drift_check_on_latest_batch():
    ds = make_dataset(generator="churn")
    client.post("/validation/run", data={"dataset_id": ds, "inject": "distribution_shift", "seed": 1}, headers=A)
    drift = client.post("/drift/check", json={"dataset_id": ds}, headers=A).json()
    assert {f["feature"] for f in drift["features"] if f["drifted"]} == {"income"}


def test_csv_upload_masks_pii():
    csv = "email,amount,label\na@x.com,1.0,0\nb@x.com,2.0,1\nc@x.com,3.0,0\nd@x.com,4.0,1\n"
    r = client.post("/datasets", data={"name": "uploaded"}, files={"file": ("d.csv", csv)}, headers=A)
    assert r.status_code == 200, r.text
    assert r.json()["profile"]["columns"]["email"]["unique"] == 1


def test_tenant_isolation():
    ds = make_dataset(headers=A)
    assert client.get(f"/datasets/{ds}/health", headers=B).status_code == 404
    assert all(a["dataset_id"] != ds for a in client.get("/alerts", headers=B).json())


def test_demo_run_end_to_end():
    r = client.post("/demo/run", json={"generator": "fraud", "failure": "schema_change"}, headers=A).json()
    assert r["report"]["root_cause"] == "schema_change"
    assert r["report"]["source"] == "rules"


def test_metrics_endpoint():
    body = client.get("/metrics").text
    assert "datasentinel_alerts_total" in body
    assert "datasentinel_dataset_health" in body


def test_audit_log_records_mutations():
    make_dataset()
    log = client.get("/audit", headers=A).json()
    assert any(entry["action"] == "create_dataset" for entry in log)
