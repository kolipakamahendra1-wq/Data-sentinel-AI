"""Per-column statistics and schema snapshot for a dataframe."""
import pandas as pd


def profile(df: pd.DataFrame) -> dict:
    columns = {}
    for name, s in df.items():
        numeric = pd.api.types.is_numeric_dtype(s)
        col = {"dtype": "numeric" if numeric else "categorical",
               "missing_frac": float(s.isna().mean()),
               "unique": int(s.nunique())}
        if numeric and s.notna().any():
            col.update(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()))
        columns[name] = col
    return {"n_rows": len(df), "duplicate_frac": float(df.duplicated().mean()) if len(df) else 0.0,
            "columns": columns}
