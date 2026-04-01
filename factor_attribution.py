from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

@dataclass
class FactorAttributionConfig:
    """Configuration for factor attribution module."""

    macro_path: str = "data/macro_factors.csv"
    regime_path: str = "data/regime_labels.csv"
    output_regression: str = "results/regression_summary.csv"
    output_shap: str = "results/shap_values.csv"
    weekly_rule: str = "W"
    rf_estimators: int = 200
    rf_random_state: int = 42

def load_csv_with_date(path: str) -> pd.DataFrame:
    """Load CSV and normalize date column/index to a datetime column named 'date'."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

def merge_data(macro_path: str, regime_path: str) -> pd.DataFrame:
    """Merge macro factors and regime labels on date."""
    macro = load_csv_with_date(macro_path)
    regime = load_csv_with_date(regime_path)
    merged = pd.merge(macro, regime, on="date", how="inner").sort_values("date")
    return merged

def compute_weekly_anomaly_frequency(df: pd.DataFrame, weekly_rule: str) -> pd.DataFrame:
    """Compute weekly anomaly_frequency and aggregate macro factors to weekly mean."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    weekly_anom = df["anomaly_flag"].resample(weekly_rule).mean().rename("anomaly_frequency")
    macro_cols = [c for c in df.columns if c not in {"regime_hmm", "regime_cusum", "anomaly_flag"}]
    weekly_macro = df[macro_cols].resample(weekly_rule).mean()

    weekly = pd.concat([weekly_anom, weekly_macro], axis=1).dropna().reset_index()
    return weekly

def run_ols(weekly: pd.DataFrame) -> pd.DataFrame:
    """Run OLS regression: anomaly_frequency ~ macro_factors."""
    y = weekly["anomaly_frequency"]
    X = weekly.drop(columns=["date", "anomaly_frequency"])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    params = model.params
    pvals = model.pvalues
    conf = model.conf_int(alpha=0.05)
    conf.columns = ["ci_lower", "ci_upper"]

    summary = pd.DataFrame(
        {
            "coef": params,
            "p_value": pvals,
            "ci_lower": conf["ci_lower"],
            "ci_upper": conf["ci_upper"],
        }
    )
    summary.index.name = "feature"
    return summary.reset_index()

def compute_shap(weekly: pd.DataFrame, n_estimators: int, random_state: int) -> pd.DataFrame:
    """Compute SHAP values using RandomForestRegressor."""
    y = weekly["anomaly_frequency"]
    X = weekly.drop(columns=["date", "anomaly_frequency"])

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    explainer = shap.Explainer(rf, X)
    shap_values = explainer(X).values

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df.insert(0, "date", weekly["date"].values)
    return shap_df

def ensure_output_dir(path: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def run_pipeline(config: FactorAttributionConfig) -> Tuple[str, str]:
    """Run factor attribution pipeline and return output paths."""
    merged = merge_data(config.macro_path, config.regime_path)
    weekly = compute_weekly_anomaly_frequency(merged, config.weekly_rule)

    reg_summary = run_ols(weekly)
    shap_df = compute_shap(weekly, config.rf_estimators, config.rf_random_state)

    ensure_output_dir(config.output_regression)
    ensure_output_dir(config.output_shap)

    reg_summary.to_csv(config.output_regression, index=False)
    shap_df.to_csv(config.output_shap, index=False)

    return config.output_regression, config.output_shap

if __name__ == "__main__":
    cfg = FactorAttributionConfig()
    reg_out, shap_out = run_pipeline(cfg)
    print(f"Saved regression summary to {reg_out}")
    print(f"Saved SHAP values to {shap_out}")
