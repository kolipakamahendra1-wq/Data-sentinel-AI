# PRD — DataSentinel AI

**AI Data Quality & ML Drift Agent**

## Table of Contents

1. [Product](#1-product)
2. [Goal](#2-goal)
3. [Problem](#3-problem)
4. [Target Users](#4-target-users)
5. [Workflow](#5-workflow)
6. [Monitoring](#6-monitoring)
7. [AI Investigation](#7-ai-investigation)
8. [Tech Stack](#8-tech-stack)
9. [Architecture](#9-architecture)
10. [Synthetic Data](#10-synthetic-data)
11. [Evaluation](#11-evaluation)
12. [Security](#12-security)
13. [Dashboard](#13-dashboard)
14. [API](#14-api)
15. [Repository](#15-repository)
16. [Delivery](#16-delivery)
17. [Portfolio Demo](#17-portfolio-demo)
18. [Recruiter Signal](#18-recruiter-signal)

---

## 1. Product

**Name:** DataSentinel AI

An AI-powered data-quality and ML-monitoring system that detects schema changes, data-quality problems, distribution drift, and model-performance degradation, then explains the likely business impact.

## 2. Goal

Create a strong ML/AI Engineer portfolio project focused on production ML reliability rather than another LLM chatbot.

## 3. Problem

ML systems can silently degrade when:

- Input distributions change
- Features become missing
- Schemas change
- Data pipelines break
- Class balance changes
- Model performance drops

Traditional monitoring detects anomalies but often does not explain them clearly.

DataSentinel combines statistical monitoring with an AI investigation layer.

## 4. Target Users

- ML Engineers
- Data Scientists
- AI Engineers
- MLOps Engineers
- Engineering teams

## 5. Workflow

```text
Data ingestion
  → Validation
  → Profiling
  → Drift detection
  → Anomaly detection
  → Model-performance check
  → AI investigation
  → Impact explanation
  → Recommended action
  → Incident report
```

## 6. Monitoring

### Data Quality

- Missing values
- Null spikes
- Duplicate records
- Invalid ranges
- Schema changes
- Cardinality changes

### Drift

- PSI
- KS test
- Population distribution comparison
- Feature drift
- Label drift

### Model

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Calibration
- Error distribution

## 7. AI Investigation

When an alert occurs, the agent should inspect:

- Recent data changes
- Pipeline events
- Feature statistics
- Model metrics
- Deployment history
- Previous incidents

Then produce a report in this shape:

> **Alert:** Feature `income` missingness increased from 1.2% to 27%.
>
> **Likely cause:** Upstream schema change.
>
> **Evidence:**
>
> - Schema version changed
> - Missingness increased immediately after deployment
> - Other features unaffected
>
> **Impact:** Model confidence decreased.
>
> **Recommended action:** Rollback or update feature transformation.

## 8. Tech Stack

| Layer          | Technologies                                  |
| -------------- | --------------------------------------------- |
| ML             | Python, Pandas, NumPy, Scikit-learn, SciPy    |
| AI             | Claude API, LangGraph, Pydantic               |
| Backend        | FastAPI                                       |
| Storage        | PostgreSQL, Parquet, MinIO/S3                 |
| Monitoring     | Evidently, Prometheus, Grafana, OpenTelemetry |
| Frontend       | React, TypeScript, Tailwind CSS               |
| Infrastructure | Docker, GitHub Actions, Cloud deployment      |

## 9. Architecture

```text
Data Sources
  → Data Profiler
  → Quality Engine
  → Drift Engine
  → Model Metrics
  → Alert Manager
  → AI Investigation Agent
  → Evidence Store
  → Dashboard
```

## 10. Synthetic Data

Generate datasets for:

- Customer churn
- Fraud detection
- Loan risk
- Product demand

Inject controlled failures:

- Missing values
- Feature distribution shifts
- Label imbalance
- Schema changes
- Outliers
- Data leakage

## 11. Evaluation

Create known failure scenarios.

Measure:

- Drift detection accuracy
- False-positive rate
- Root-cause identification
- Explanation correctness
- Alert-to-report latency

## 12. Security

- Dataset access controls
- API authentication
- PII masking
- Audit logs
- Tenant isolation

## 13. Dashboard

Show:

- Dataset health
- Drift score
- Feature health
- Model health
- Active alerts
- Historical incidents
- AI explanation
- Recommended action

## 14. API

| Method | Endpoint                |
| ------ | ----------------------- |
| `POST` | `/datasets`             |
| `POST` | `/validation/run`       |
| `GET`  | `/datasets/{id}/health` |
| `POST` | `/drift/check`          |
| `GET`  | `/alerts`               |
| `GET`  | `/alerts/{id}`          |
| `POST` | `/investigations`       |
| `GET`  | `/metrics`              |

## 15. Repository

```text
datasentinel-ai/
├── backend/
├── monitoring/
├── drift/
├── profiling/
├── agents/
├── api/
├── ml/
├── data/
├── frontend/
├── evaluation/
├── tests/
├── infrastructure/
├── .github/
│   └── workflows/
├── README.md
└── PRD.md
```

## 16. Delivery

| Phase | Scope                        |
| ----- | ---------------------------- |
| 1     | Data generator + profiler    |
| 2     | Quality and schema checks    |
| 3     | Drift detection              |
| 4     | Model monitoring             |
| 5     | AI investigation agent       |
| 6     | Dashboard                    |
| 7     | Evaluation + observability   |
| 8     | Docker + cloud + CI/CD       |

## 17. Portfolio Demo

Inject a controlled data failure, then show:

```text
Normal data
  → Deployment
  → Drift
  → Alert
  → AI investigation
  → Evidence
  → Root cause
  → Recommended fix
```

## 18. Recruiter Signal

Demonstrates: Python + ML + statistics + production monitoring + AI agents + MLOps + observability.

This is deliberately different from a normal RAG application.
