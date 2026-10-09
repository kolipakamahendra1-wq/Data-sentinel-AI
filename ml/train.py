"""Train the baseline model on reference data."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.model_metrics import evaluate


def train(df: pd.DataFrame, label: str = "label"):
    """Fit on 75% of the reference data; return (model, baseline metrics on the 25% holdout)."""
    X = df.drop(columns=[label])
    numeric = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X if c not in numeric and X[c].nunique() < 0.5 * len(X)]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    train_df, hold = train_test_split(df, test_size=0.25, random_state=0, stratify=df[label])
    model.fit(train_df[numeric + categorical], train_df[label])
    return model, evaluate(model, hold, label)
