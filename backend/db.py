"""SQLite metadata store + Parquet/joblib files on disk, all under DATASENTINEL_STORAGE."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STORAGE = Path(os.getenv("DATASENTINEL_STORAGE", "storage"))
STORAGE.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY, tenant_id TEXT, name TEXT, generator TEXT,
    profile TEXT, baseline TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, dataset_id INTEGER, batch_path TEXT, result TEXT,
    health_score INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, tenant_id TEXT, dataset_id INTEGER, run_id INTEGER,
    title TEXT, severity TEXT, status TEXT, report TEXT, latency_ms REAL, created_at TEXT, investigated_at TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, dataset_id INTEGER, type TEXT, message TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, tenant_id TEXT, action TEXT, detail TEXT, created_at TEXT);
"""

JSON_COLUMNS = {"profile", "baseline", "result", "report", "detail"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(STORAGE / "datasentinel.db")
    conn.row_factory = sqlite3.Row
    return conn


with connect() as _c:
    _c.executescript(SCHEMA)


def _decode(row: sqlite3.Row) -> dict:
    return {k: json.loads(row[k]) if k in JSON_COLUMNS and row[k] is not None else row[k] for k in row.keys()}


def query(sql: str, *args) -> list[dict]:
    with connect() as conn:
        return [_decode(r) for r in conn.execute(sql, args).fetchall()]


def execute(sql: str, *args) -> int:
    args = tuple(json.dumps(a, default=str) if isinstance(a, (dict, list)) else a for a in args)
    with connect() as conn:
        return conn.execute(sql, args).lastrowid


def dataset_dir(dataset_id: int) -> Path:
    path = STORAGE / "datasets" / str(dataset_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def audit(tenant: str, action: str, detail: dict) -> None:
    execute("INSERT INTO audit_log (tenant_id, action, detail, created_at) VALUES (?,?,?,?)", tenant, action, detail, now())
