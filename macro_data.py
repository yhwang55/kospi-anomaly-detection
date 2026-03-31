from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred

@dataclass
class MacroConfig:
    """Configuration for macro data pipeline."""

    start_date: str = "2020-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")
    fred_series: dict[str, str] = field(
        default_factory=lambda: {
            "us_10y": "DGS10",          # US 10Y Treasury
            "usd_krw": "DEXKOUS",       # USD/KRW
            "vix": "VIXCLS",            # VIX
            "inflation_exp": "T10YIE",  # US 10Y Breakeven Inflation
        }
    )
    ecos_api_key_env: str = "ECOS_API_KEY"
    fred_api_key_env: str = "FRED_API_KEY"
    output_path: str = "data/macro_factors.csv"


def load_env() -> None:
    """Load environment variables from .env."""
    load_dotenv(override=False)


def get_fred_series(
    fred: Fred, series_id: str, start_date: str, end_date: str
) -> pd.Series:
    """Fetch a FRED series and return as a named pandas Series."""
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    s = pd.Series(data)
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def fetch_bok_base_rate_ecos(
    api_key: str, start_date: str, end_date: str
) -> Optional[pd.Series]:
    """
    Fetch BOK base rate from ECOS API.
    Uses the commonly referenced ECOS statistics:
      - STAT_CODE: 722Y001 (Base rate)
      - ITEM_CODE1: 0101000
      - CYCLE: M (monthly)

    Returns a monthly Series with datetime index, or None on failure.
    """
    # ECOS API date format: YYYYMMDD
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    stat_code = "722Y001"
    cycle = "M"
    item_code1 = "0101000"

    url = (
        "https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{api_key}/json/kr/1/10000/"
        f"{stat_code}/{cycle}/{start}/{end}/{item_code1}"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return None

        dates = []
        values = []
        for r in rows:
            # TIME format for M: YYYYMM
            time_str = r.get("TIME", "")
            if len(time_str) == 6:
                dt = datetime.strptime(time_str + "01", "%Y%m%d")
                dates.append(dt)
                values.append(float(r.get("DATA_VALUE", "nan")))

        s = pd.Series(values, index=pd.to_datetime(dates), name="bok_rate")
        return s.sort_index()
    except Exception:
        return None


def fetch_bok_base_rate_scrape(start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Fallback scrape for BOK base rate if ECOS API is unavailable.
    Tries to parse a BOK base rate history table via pandas.read_html.

    Returns a Series or None on failure.
    """
    # This URL has historically contained a base rate history table.
    # If it changes, scraping may fail gracefully.
    url = "https://www.bok.or.kr/eng/singl/monetarypolicy/baseRate.do?menuNo=400052"

    try:
        tables = pd.read_html(url)
        if not tables:
            return None

        # Heuristic: find a table with a "Date" column and a "Base Rate" column.
        for tbl in tables:
            cols = [c.lower() for c in tbl.columns.astype(str)]
            if any("date" in c for c in cols) and any("base" in c for c in cols):
                df = tbl.copy()
                # Attempt to normalize column names
                date_col = next(c for c in df.columns if "date" in str(c).lower())
                rate_col = next(c for c in df.columns if "base" in str(c).lower())

                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")

                df = df.dropna(subset=[date_col, rate_col])
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

                s = pd.Series(df[rate_col].values, index=df[date_col], name="bok_rate")
                return s.sort_index()

        return None
    except Exception:
        return None


def get_bok_base_rate(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch BOK base rate using ECOS API if available,
    otherwise attempt a scrape fallback.

    Returns a Series (may be empty if both methods fail).
    """
    api_key = os.getenv("ECOS_API_KEY")
    if api_key:
        series = fetch_bok_base_rate_ecos(api_key, start_date, end_date)
        if series is not None and not series.empty:
            return series

    series = fetch_bok_base_rate_scrape(start_date, end_date)
    if series is not None and not series.empty:
        return series

    # Return empty series if all methods fail
    return pd.Series(dtype=float, name="bok_rate")


def build_macro_factors(config: MacroConfig) -> pd.DataFrame:
    """
    Build a merged macro factors DataFrame with daily frequency and forward-fill.
    """
    fred_key = os.getenv(config.fred_api_key_env)
    if not fred_key:
        raise ValueError(
            f"Missing {config.fred_api_key_env} in environment or .env file."
        )

    fred = Fred(api_key=fred_key)

    # Fetch FRED series
    series_list = []
    for col_name, series_id in config.fred_series.items():
        s = get_fred_series(fred, series_id, config.start_date, config.end_date)
        s = s.rename(col_name)
        series_list.append(s)

    # Fetch BOK base rate (monthly) and include
    bok = get_bok_base_rate(config.start_date, config.end_date)
    if not bok.empty:
        series_list.append(bok)

    # Merge on date index
    df = pd.concat(series_list, axis=1)

    # Ensure bok_rate column exists
    if "bok_rate" not in df.columns:
        print("⚠️  Warning: bok_rate unavailable, filling with NaN")
        df["bok_rate"] = float("nan")

    # Reindex to daily frequency and forward-fill
    full_index = pd.date_range(config.start_date, config.end_date, freq="D")
    df = df.reindex(full_index)
    df = df.sort_index().ffill()

    df.index.name = "date"
    return df


def save_macro_factors(df: pd.DataFrame, output_path: str) -> None:
    """Save macro factors DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)


def run_pipeline() -> str:
    """Run the full macro data pipeline and return output path."""
    load_env()
    config = MacroConfig()
    df = build_macro_factors(config)
    save_macro_factors(df, config.output_path)
    return config.output_path


if __name__ == "__main__":
    output = run_pipeline()
    print(f"Saved macro factors to {output}")
