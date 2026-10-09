"""Synthetic datasets with a binary `label` column, plus controlled failure injection."""
import numpy as np
import pandas as pd

DATASETS = ["churn", "fraud", "loan", "demand"]
KEY_FEATURE = {"churn": "income", "fraud": "amount", "loan": "income", "demand": "price"}
FAILURES = ["missing_values", "distribution_shift", "label_imbalance",
            "schema_change", "outliers", "data_leakage"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _churn(rng, n):
    tenure = rng.integers(1, 73, n)
    charges = rng.normal(70, 20, n)
    calls = rng.poisson(2, n)
    contract = rng.choice(["monthly", "one_year", "two_year"], n, p=[0.5, 0.3, 0.2])
    income = rng.normal(55000, 15000, n).clip(8000)
    logit = (-1 + 0.03 * (charges - 70) + 0.4 * (calls - 2) - 0.03 * (tenure - 36)
             + 0.8 * (contract == "monthly") - 0.00002 * (income - 55000))
    return pd.DataFrame({"email": [f"user{i}@example.com" for i in range(n)],
                         "tenure_months": tenure, "monthly_charges": charges, "support_calls": calls,
                         "contract": contract, "income": income}), logit


def _fraud(rng, n):
    amount = rng.lognormal(4, 1, n)
    hour = rng.integers(0, 24, n)
    merchant = rng.choice(["grocery", "travel", "electronics", "online"], n)
    age = rng.integers(1, 3650, n)
    income = rng.normal(50000, 15000, n).clip(8000)
    logit = (-3 + 0.8 * (np.log(amount) - 4) + 0.8 * (hour < 6) + 1.0 * (merchant == "online")
             - 0.0005 * (age - 1800))
    return pd.DataFrame({"amount": amount, "hour": hour, "merchant_category": merchant,
                         "account_age_days": age, "cardholder_income": income}), logit


def _loan(rng, n):
    income = rng.normal(60000, 20000, n).clip(8000)
    loan = rng.normal(15000, 6000, n).clip(500)
    score = rng.normal(680, 60, n)
    employment = rng.choice(["salaried", "self_employed", "unemployed"], n, p=[0.7, 0.2, 0.1])
    debt = rng.beta(2, 5, n)
    logit = (-1.5 - 0.02 * (score - 680) + 3 * (debt - 0.29) + 1.2 * (employment == "unemployed")
             + 0.00003 * (loan - 15000) - 0.00002 * (income - 60000))
    return pd.DataFrame({"email": [f"applicant{i}@example.com" for i in range(n)],
                         "income": income, "loan_amount": loan, "credit_score": score,
                         "employment_type": employment, "debt_ratio": debt}), logit


def _demand(rng, n):
    price = rng.normal(20, 5, n)
    promo = rng.choice(["none", "discount", "bundle"], n, p=[0.6, 0.25, 0.15])
    temp = rng.normal(20, 8, n)
    dow = rng.integers(0, 7, n)
    income = rng.normal(52000, 12000, n).clip(8000)
    logit = -0.2 * (price - 20) + 1.0 * (promo != "none") + 0.05 * (temp - 20) + 0.5 * (dow >= 5) - 0.5
    return pd.DataFrame({"price": price, "promo": promo, "temperature": temp,
                         "day_of_week": dow, "regional_income": income}), logit


_BUILDERS = {"churn": _churn, "fraud": _fraud, "loan": _loan, "demand": _demand}


def generate(name: str, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df, logit = _BUILDERS[name](rng, n)
    df["label"] = (rng.random(n) < _sigmoid(logit)).astype(int)
    return df


def inject(df: pd.DataFrame, kind: str, feature: str | None = None, seed: int = 0) -> pd.DataFrame:
    """Return a copy of `df` with one controlled failure applied to `feature`."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    if feature is None:
        feature = next(f for f in KEY_FEATURE.values() if f in df.columns)
    col = df[feature]
    if kind == "missing_values":
        df.loc[rng.random(len(df)) < 0.3, feature] = np.nan
    elif kind == "distribution_shift":
        df[feature] = col + 2 * col.std()
    elif kind == "label_imbalance":
        neg = df.index[df["label"] == 0]
        drop = rng.choice(neg, size=int(len(neg) * 0.85), replace=False)
        df = df.drop(index=drop).reset_index(drop=True)
    elif kind == "schema_change":
        df = df.rename(columns={feature: f"{feature}_v2"})
    elif kind == "outliers":
        df[feature] = col.astype(float)
        df.loc[rng.random(len(df)) < 0.05, feature] = col.mean() + 20 * col.std()
    elif kind == "data_leakage":
        df["approval_score"] = df["label"] + rng.normal(0, 0.05, len(df))
    else:
        raise ValueError(f"unknown failure kind: {kind}")
    return df
