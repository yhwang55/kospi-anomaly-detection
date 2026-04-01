from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf


@dataclass
class TableauExportConfig:
    """Configuration for Tableau export module."""

    macro_path: str = "data/macro_factors.csv"
    regime_path: str = "data/regime_labels.csv"
    output_path: str = "tableau/kospi_dashboard_data.csv"
    kospi_ticker: str = "^KS11"
    start_date: str = "2020-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")


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


def fetch_kospi_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch KOSPI index prices and compute daily returns."""
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "kospi_price", "kospi_return"])

    close = df["Close"]
    if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
        close = close.squeeze(axis=1)

    out = pd.DataFrame(
        {
            "date": close.index,
            "kospi_price": close.values,
        }
    )
    out["kospi_return"] = out["kospi_price"].pct_change()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def build_tableau_dataset(config: TableauExportConfig) -> pd.DataFrame:
    """Build a flat dataset for Tableau dashboard."""
    macro = load_csv_with_date(config.macro_path)
    regime = load_csv_with_date(config.regime_path)
    kospi = fetch_kospi_prices(config.kospi_ticker, config.start_date, config.end_date)

    merged = pd.merge(kospi, regime, on="date", how="left")
    merged = pd.merge(merged, macro, on="date", how="left")
    merged = merged.sort_values("date")

    merged["anomaly_flag"] = pd.to_numeric(merged.get("anomaly_flag"), errors="coerce")
    merged["anomaly_frequency_weekly"] = merged["anomaly_flag"].rolling(5).mean()

    merged = merged.ffill()
    merged = merged.dropna(subset=["kospi_price"])

    columns = [
        "date",
        "kospi_price",
        "kospi_return",
        "anomaly_flag",
        "regime_hmm",
        "regime_cusum",
        "vix",
        "usd_krw",
        "us_10y",
        "inflation_exp",
        "bok_rate",
        "anomaly_frequency_weekly",
    ]

    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged[columns]


def save_tableau_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save Tableau dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def run_pipeline() -> str:
    """Run the Tableau export pipeline and return output path."""
    config = TableauExportConfig()
    df = build_tableau_dataset(config)
    save_tableau_dataset(df, config.output_path)
    return config.output_path


if __name__ == "__main__":
    output = run_pipeline()
    print(f"Saved Tableau dashboard dataset to {output}")