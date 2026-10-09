"""DataSentinel AI REST API."""
import io
import os
import time

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from agents.investigator import investigate
from backend import db
from data.generator import DATASETS, FAILURES, generate, inject
from drift.drift import check_drift
from ml.train import train
from monitoring.checks import run_checks
from monitoring.quality import mask_pii
from profiling.profiler import profile

app = FastAPI(title="DataSentinel AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALERTS = Counter("datasentinel_alerts_total", "Alerts raised", ["severity"])
HEALTH = Gauge("datasentinel_dataset_health", "Latest dataset health score (0-100)", ["dataset"])
MAX_PSI = Gauge("datasentinel_max_psi", "Latest max feature PSI", ["dataset"])
INVESTIGATION_SECONDS = Histogram("datasentinel_investigation_seconds", "Alert investigation latency", ["source"],
                                  buckets=(0.05, 0.5, 2, 5, 15, 30, 60, 120))

API_KEYS = dict(pair.split(":", 1) for pair in os.getenv("DATASENTINEL_API_KEYS", "demo-key:demo").split(","))


def tenant(x_api_key: str | None = Header(default=None)) -> str:
    if x_api_key not in API_KEYS:
        raise HTTPException(401, "invalid or missing X-API-Key")
    return API_KEYS[x_api_key]


def get_dataset(dataset_id: int, tenant_id: str) -> dict:
    rows = db.query("SELECT * FROM datasets WHERE id=? AND tenant_id=?", dataset_id, tenant_id)
    if not rows:
        raise HTTPException(404, "dataset not found")
    return rows[0]


def read_csv(file: UploadFile) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file.file.read()))


@app.post("/datasets")
def create_dataset(name: str = Form(...), generator: str | None = Form(None), n: int = Form(2000),
                   seed: int = Form(0), file: UploadFile | None = File(None), t: str = Depends(tenant)):
    if file is not None:
        df = read_csv(file)
    elif generator in DATASETS:
        df = generate(generator, n, seed)
    else:
        raise HTTPException(422, f"provide a CSV file or generator in {DATASETS}")
    df = mask_pii(df)
    model, baseline = (train(df) if "label" in df and df["label"].nunique() == 2 and len(df) >= 100 else (None, None))
    ds_id = db.execute("INSERT INTO datasets (tenant_id, name, generator, profile, baseline, created_at) VALUES (?,?,?,?,?,?)",
                       t, name, generator, profile(df), baseline, db.now())
    path = db.dataset_dir(ds_id)
    df.to_parquet(path / "reference.parquet")
    if model is not None:
        joblib.dump(model, path / "model.joblib")
    db.audit(t, "create_dataset", {"dataset_id": ds_id, "name": name, "rows": len(df)})
    return get_dataset(ds_id, t)


@app.get("/datasets")
def list_datasets(t: str = Depends(tenant)):
    return db.query("""SELECT d.id, d.name, d.generator, d.created_at,
                       (SELECT health_score FROM runs r WHERE r.dataset_id=d.id ORDER BY r.id DESC LIMIT 1) AS health_score
                       FROM datasets d WHERE tenant_id=? ORDER BY d.id DESC""", t)


@app.post("/validation/run")
def run_validation(dataset_id: int = Form(...), inject_failure: str | None = Form(None, alias="inject"),
                   seed: int = Form(1), file: UploadFile | None = File(None), t: str = Depends(tenant)):
    ds = get_dataset(dataset_id, t)
    if file is not None:
        cur = read_csv(file)
    elif ds["generator"]:
        cur = generate(ds["generator"], ds["profile"]["n_rows"], seed)
    else:
        raise HTTPException(422, "upload a CSV batch for this dataset")
    if inject_failure:
        if inject_failure not in FAILURES:
            raise HTTPException(422, f"inject must be one of {FAILURES}")
        cur = inject(cur, inject_failure)
    cur = mask_pii(cur)

    path = db.dataset_dir(dataset_id)
    ref = pd.read_parquet(path / "reference.parquet")
    model = joblib.load(path / "model.joblib") if (path / "model.joblib").exists() else None
    result = run_checks(ref, cur, model, ds["baseline"])

    batch_path = path / f"batch_{time.time_ns()}.parquet"
    cur.to_parquet(batch_path)
    run_id = db.execute("INSERT INTO runs (dataset_id, batch_path, result, health_score, created_at) VALUES (?,?,?,?,?)",
                        dataset_id, str(batch_path), result, result["health_score"], db.now())
    alert_id = None
    if result["alert"]:
        alert_id = db.execute("""INSERT INTO alerts (tenant_id, dataset_id, run_id, title, severity, status, created_at)
                                 VALUES (?,?,?,?,?,'open',?)""", t, dataset_id, run_id, result["title"], result["severity"], db.now())
        ALERTS.labels(result["severity"]).inc()
    HEALTH.labels(ds["name"]).set(result["health_score"])
    MAX_PSI.labels(ds["name"]).set(result["drift"]["max_psi"])
    db.audit(t, "run_validation", {"dataset_id": dataset_id, "run_id": run_id, "inject": inject_failure})
    return {"run_id": run_id, "alert_id": alert_id, **result}


def latest_run(dataset_id: int) -> dict:
    rows = db.query("SELECT * FROM runs WHERE dataset_id=? ORDER BY id DESC LIMIT 1", dataset_id)
    if not rows:
        raise HTTPException(404, "no validation runs yet")
    return rows[0]


@app.get("/datasets/{dataset_id}/health")
def dataset_health(dataset_id: int, t: str = Depends(tenant)):
    ds = get_dataset(dataset_id, t)
    run = latest_run(dataset_id)
    r = run["result"]
    return {"dataset_id": dataset_id, "name": ds["name"], "run_id": run["id"], "checked_at": run["created_at"],
            "health_score": r["health_score"], "feature_health": r["feature_health"], "drift": r["drift"],
            "model": r["model"], "baseline": ds["baseline"], "findings": r["findings"]}


class DriftRequest(BaseModel):
    dataset_id: int


@app.post("/drift/check")
def drift_check(req: DriftRequest, t: str = Depends(tenant)):
    get_dataset(req.dataset_id, t)
    ref = pd.read_parquet(db.dataset_dir(req.dataset_id) / "reference.parquet")
    return check_drift(ref, pd.read_parquet(latest_run(req.dataset_id)["batch_path"]))


class EventRequest(BaseModel):
    dataset_id: int
    type: str
    message: str


@app.post("/events")
def create_event(req: EventRequest, t: str = Depends(tenant)):
    get_dataset(req.dataset_id, t)
    event_id = db.execute("INSERT INTO events (dataset_id, type, message, created_at) VALUES (?,?,?,?)",
                          req.dataset_id, req.type, req.message, db.now())
    db.audit(t, "create_event", req.model_dump())
    return {"id": event_id}


@app.get("/alerts")
def list_alerts(status: str | None = None, t: str = Depends(tenant)):
    sql = "SELECT * FROM alerts WHERE tenant_id=?" + (" AND status=?" if status else "") + " ORDER BY id DESC"
    return db.query(sql, *((t, status) if status else (t,)))


def get_alert(alert_id: int, tenant_id: str) -> dict:
    rows = db.query("SELECT * FROM alerts WHERE id=? AND tenant_id=?", alert_id, tenant_id)
    if not rows:
        raise HTTPException(404, "alert not found")
    return rows[0]


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: int, t: str = Depends(tenant)):
    alert = get_alert(alert_id, t)
    alert["run"] = db.query("SELECT result FROM runs WHERE id=?", alert["run_id"])[0]["result"]
    return alert


class InvestigationRequest(BaseModel):
    alert_id: int
    use_llm: bool = True


@app.post("/investigations")
def create_investigation(req: InvestigationRequest, t: str = Depends(tenant)):
    alert = get_alert(req.alert_id, t)
    check = db.query("SELECT result FROM runs WHERE id=?", alert["run_id"])[0]["result"]
    context = {
        "events": db.query("SELECT type, message, created_at FROM events WHERE dataset_id=? ORDER BY id DESC LIMIT 5",
                           alert["dataset_id"]),
        "previous_incidents": [{"title": a["title"], "root_cause": a["report"]["root_cause"], "at": a["created_at"]}
                               for a in db.query("""SELECT title, report, created_at FROM alerts WHERE dataset_id=?
                                                    AND report IS NOT NULL AND id<>? ORDER BY id DESC LIMIT 3""",
                                                 alert["dataset_id"], alert["id"])],
    }
    start = time.perf_counter()
    report = investigate(check, context, use_llm=req.use_llm)
    latency = time.perf_counter() - start
    INVESTIGATION_SECONDS.labels(report.source).observe(latency)
    db.execute("UPDATE alerts SET status='investigated', report=?, latency_ms=?, investigated_at=? WHERE id=?",
               report.model_dump(), round(latency * 1000, 1), db.now(), alert["id"])
    db.audit(t, "investigate", {"alert_id": alert["id"], "source": report.source})
    return {**report.model_dump(), "latency_ms": round(latency * 1000, 1)}


class DemoRequest(BaseModel):
    generator: str = "loan"
    failure: str = "missing_values"
    use_llm: bool = True


@app.post("/demo/run")
def demo_run(req: DemoRequest, t: str = Depends(tenant)):
    """Portfolio demo: normal data -> deployment -> injected failure -> alert -> AI investigation."""
    if req.generator not in DATASETS or req.failure not in FAILURES:
        raise HTTPException(422, f"generator in {DATASETS}, failure in {FAILURES}")
    ds = create_dataset(name=f"demo-{req.generator}", generator=req.generator, n=2000, seed=0, file=None, t=t)
    normal = run_validation(dataset_id=ds["id"], inject_failure=None, seed=1, file=None, t=t)
    create_event(EventRequest(dataset_id=ds["id"], type="deployment",
                              message="feature-pipeline v2.3 deployed (schema version 7 -> 8)"), t)
    failed = run_validation(dataset_id=ds["id"], inject_failure=req.failure, seed=2, file=None, t=t)
    report = (create_investigation(InvestigationRequest(alert_id=failed["alert_id"], use_llm=req.use_llm), t)
              if failed["alert_id"] else None)
    return {"dataset_id": ds["id"], "normal_health": normal["health_score"], "failed_health": failed["health_score"],
            "alert_id": failed["alert_id"], "report": report}


@app.get("/audit")
def audit_log(t: str = Depends(tenant)):
    return db.query("SELECT action, detail, created_at FROM audit_log WHERE tenant_id=? ORDER BY id DESC LIMIT 100", t)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
