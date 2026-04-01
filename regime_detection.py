from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

@dataclass
class RegimeConfig:
    """Configuration for regime detection pipeline."""

    start_date: str = "2020-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")
    top_n: int = 100
    target_stock_name: str = "Samsung Electronics"
    output_path: str = "data/regime_labels.csv"
    contamination: float = 0.05
    hmm_states: int = 2
    cusum_threshold: float = 0.02  # sensitivity for change points

def fetch_krx_listing(top_n: int = 100) -> pd.DataFrame | None:
    """
    Fetch KOSPI constituent list from KRX open data portal.
    Returns a DataFrame (top_n by market cap) or None if fetch fails.
    """
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
        "mktId": "STK",
        "trdDd": "20241231",
        "money": "1",
        "csvxls_isNo": "false",
    }
    headers = {"Referer": "http://data.krx.co.kr/"}

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        raw = resp.json()["OutBlock_1"]
        df = pd.DataFrame(raw)
        df = df[["ISU_SRT_CD", "ISU_ABBRV", "MKTCAP"]].rename(
            columns={"ISU_SRT_CD": "ticker_krx", "ISU_ABBRV": "name", "MKTCAP": "mktcap_krw"}
        )
        df["mktcap_krw"] = pd.to_numeric(
            df["mktcap_krw"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.sort_values("mktcap_krw", ascending=False).head(top_n).reset_index(drop=True)
        return df
    except Exception:
        return None

def build_ticker_map(krx_df: pd.DataFrame | None) -> Dict[str, str]:
    """
    Build ticker map: {yfinance_ticker: name}.
    Falls back to a representative subset if KRX fetch fails.
    """
    if krx_df is not None:
        return {f"{row['ticker_krx']}.KS": row["name"] for _, row in krx_df.iterrows()}

    fallback = {
        "005930": "Samsung Electronics",
        "000660": "SK Hynix",
        "035420": "NAVER",
        "005380": "Hyundai Motor",
        "051910": "LG Chem",
        "000270": "Kia",
        "068270": "Celltrion",
        "028260": "Samsung C&T",
        "105560": "KB Financial",
        "055550": "Shinhan Financial",
        "012330": "Hyundai Mobis",
        "207940": "Samsung Biologics",
        "006400": "Samsung SDI",
        "066570": "LG Electronics",
        "003550": "LG Corp",
        "032830": "Samsung Life",
        "017670": "SK Telecom",
        "030200": "KT Corp",
        "096770": "SK Innovation",
        "011200": "HMM",
    }
    return {f"{code}.KS": name for code, name in fallback.items()}

def download_ohlcv(
    tickers: List[str], start_date: str, end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download OHLCV data from yfinance.
    Returns (close_df, volume_df).
    """
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)
    close = raw["Close"]
    volume = raw["Volume"]
    return close, volume

def filter_valid_tickers(
    close: pd.DataFrame, volume: pd.DataFrame, max_missing: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop tickers with missing data above max_missing threshold.
    """
    missing_pct = close.isnull().mean()
    valid = missing_pct[missing_pct < max_missing].index.tolist()
    return close[valid], volume[valid]

def build_features(close_df: pd.DataFrame, volume_df: pd.DataFrame, ticker_name: str) -> pd.DataFrame:
    """
    Build anomaly-detection features for a single stock.
    """
    df = pd.DataFrame()
    df["Close"] = close_df[ticker_name]
    df["Volume"] = volume_df[ticker_name]

    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Price_vs_MA5"] = (df["Close"] - df["MA5"]) / df["MA5"]
    df["Price_vs_MA20"] = (df["Close"] - df["MA20"]) / df["MA20"]

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["Volume_zscore"] = (df["Volume"] - vol_mean) / vol_std

    df["Volatility_5d"] = df["Return"].rolling(5).std()
    df["PV_signal"] = df["Return"] * df["Volume_zscore"]

    return df.dropna()

def compute_stock_anomalies(df_feat: pd.DataFrame, contamination: float) -> pd.Series:
    """
    Compute Isolation Forest anomaly flag for a single stock's features.
    """
    feature_cols = [
        "Return",
        "Price_vs_MA5",
        "Price_vs_MA20",
        "Volume_zscore",
        "Volatility_5d",
        "PV_signal",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat[feature_cols])

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        max_samples="auto",
    )
    preds = model.fit_predict(X_scaled)  # -1: anomaly, 1: normal
    return pd.Series(preds == -1, index=df_feat.index, name="anomaly_flag")

def compute_market_anomaly_flag(
    close_df: pd.DataFrame, volume_df: pd.DataFrame, contamination: float, threshold: float = 0.10
) -> pd.Series:
    """
    Compute market-wide anomaly_ratio per date using all valid stocks.
    anomaly_flag = 1 if anomaly_ratio > threshold, else 0.
    """
    anomaly_flags = []

    for ticker in close_df.columns:
        df_feat = build_features(close_df, volume_df, ticker)
        stock_flag = compute_stock_anomalies(df_feat, contamination)
        anomaly_flags.append(stock_flag.rename(ticker))

    flags_df = pd.concat(anomaly_flags, axis=1).dropna(how="all")
    anomaly_ratio = flags_df.mean(axis=1)
    anomaly_flag = (anomaly_ratio > threshold).astype(int)
    anomaly_flag.name = "anomaly_flag"

    return anomaly_flag

def compute_hmm_regime(returns: pd.Series, n_states: int = 2) -> pd.Series:
    """
    Compute HMM regimes on return series.
    Returns regime labels (1 = bull, 0 = bear).
    """
    data = returns.values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(data)
    states = model.predict(data)

    # Map to bull/bear based on mean return in each state
    state_means = pd.Series(returns).groupby(states).mean()
    bull_state = state_means.idxmax()
    regime = (states == bull_state).astype(int)

    return pd.Series(regime, index=returns.index, name="regime_hmm")

def compute_cusum_regime(returns: pd.Series, threshold: float = 0.02) -> pd.Series:
    """
    Simple CUSUM change-point regime detection.
    Returns regime labels (1 = bull, 0 = bear).
    """
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    mean_return = returns.mean()
    s_pos = 0.0
    s_neg = 0.0
    regime = 1
    regimes: List[int] = []

    for r in returns:
        s_pos = max(0.0, s_pos + r - mean_return)
        s_neg = min(0.0, s_neg + r - mean_return)

        if s_pos > threshold:
            regime = 1
            s_pos = 0.0
            s_neg = 0.0
        elif abs(s_neg) > threshold:
            regime = 0
            s_pos = 0.0
            s_neg = 0.0

        regimes.append(regime)

    return pd.Series(regimes, index=returns.index, name="regime_cusum")

def fetch_kospi_index(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch KOSPI composite index (^KS11) adjusted close and return series.
    """
    kospi = yf.download("^KS11", start=start_date, end=end_date, auto_adjust=True)
    returns = kospi["Close"].pct_change().dropna()
    return returns

def build_regime_labels(config: RegimeConfig) -> pd.DataFrame:
    """
    Build regime labels DataFrame with HMM and CUSUM regimes plus anomaly flag.
    """
    krx_df = fetch_krx_listing(top_n=config.top_n)
    ticker_map = build_ticker_map(krx_df)

    close, volume = download_ohlcv(list(ticker_map.keys()), config.start_date, config.end_date)
    close = close.rename(columns=ticker_map)
    volume = volume.rename(columns=ticker_map)

    close, volume = filter_valid_tickers(close, volume)

    anomaly_flag = compute_market_anomaly_flag(
        close, volume, contamination=config.contamination, threshold=0.10
    )

    kospi_returns = fetch_kospi_index(config.start_date, config.end_date)
    regime_hmm = compute_hmm_regime(kospi_returns, n_states=config.hmm_states)
    regime_cusum = compute_cusum_regime(kospi_returns, threshold=config.cusum_threshold)

    df_out = pd.DataFrame(
        {
            "date": kospi_returns.index,
            "regime_hmm": regime_hmm.values,
            "regime_cusum": regime_cusum.values,
            "anomaly_flag": anomaly_flag.reindex(kospi_returns.index).fillna(0).astype(int).values,
        }
    )

    return df_out

def save_regime_labels(df: pd.DataFrame, output_path: str) -> None:
    """Save regime labels DataFrame to CSV."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(output_path, index=False)

def run_pipeline() -> str:
    """Run the regime detection pipeline and return output path."""
    config = RegimeConfig()
    df = build_regime_labels(config)
    save_regime_labels(df, config.output_path)
    return config.output_path

if __name__ == "__main__":
    output = run_pipeline()
    print(f"Saved regime labels to {output}")