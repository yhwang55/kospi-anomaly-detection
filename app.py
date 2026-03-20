"""
KOSPI Anomaly Detection Dashboard
Yoon Hwang — Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import shap
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KOSPI Anomaly Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label { color: #8b95b0; font-size: 13px; font-weight: 500; letter-spacing: 0.5px; }
    .metric-value { color: #e8ecf5; font-size: 28px; font-weight: 700; margin-top: 4px; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .section-header {
        color: #c5cae9;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3250;
    }
    div[data-testid="stSidebar"] { background-color: #1a1d2e; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
TICKERS = {
    '005930.KS': 'Samsung Electronics',
    '000660.KS': 'SK Hynix',
    '035420.KS': 'NAVER',
    '005380.KS': 'Hyundai Motor',
    '051910.KS': 'LG Chem',
    '000270.KS': 'Kia',
    '068270.KS': 'Celltrion',
    '028260.KS': 'Samsung C&T',
    '105560.KS': 'KB Financial',
    '055550.KS': 'Shinhan Financial',
    '012330.KS': 'Hyundai Mobis',
    '207940.KS': 'Samsung Biologics',
    '006400.KS': 'Samsung SDI',
    '066570.KS': 'LG Electronics',
    '003550.KS': 'LG Corp',
    '032830.KS': 'Samsung Life',
    '017670.KS': 'SK Telecom',
    '030200.KS': 'KT Corp',
    '096770.KS': 'SK Innovation',
    '011200.KS': 'HMM',
}

MARKET_EVENTS = pd.DataFrame([
    {"date": "2020-03-19", "event": "KOSPI 52-week low — COVID crash",               "category": "Macro",        "impact": "High"},
    {"date": "2020-11-09", "event": "Pfizer vaccine announcement",                    "category": "Macro",        "impact": "High"},
    {"date": "2021-01-11", "event": "KOSPI all-time high 3266 (Donghak Gaeami)",      "category": "Macro",        "impact": "High"},
    {"date": "2022-02-24", "event": "Russia invades Ukraine",                         "category": "Geopolitical", "impact": "High"},
    {"date": "2022-06-16", "event": "Fed +75bp — largest hike since 1994",            "category": "Macro",        "impact": "High"},
    {"date": "2023-03-10", "event": "SVB collapse",                                   "category": "Macro",        "impact": "High"},
    {"date": "2023-05-25", "event": "NVIDIA AI guidance shock — SK Hynix HBM surge",  "category": "Earnings",     "impact": "High"},
    {"date": "2023-04-07", "event": "Samsung voluntary memory production cut",         "category": "Earnings",     "impact": "High"},
    {"date": "2024-03-20", "event": "AI chip surge — Samsung/SK Hynix rally",         "category": "Earnings",     "impact": "High"},
    {"date": "2024-08-05", "event": "Yen carry trade unwind — KOSPI −8.8%",           "category": "Macro",        "impact": "High"},
    {"date": "2024-09-18", "event": "Fed first rate cut since 2020",                  "category": "Macro",        "impact": "High"},
    {"date": "2024-11-05", "event": "Trump election win",                              "category": "Geopolitical", "impact": "High"},
    {"date": "2024-12-04", "event": "Korea martial law crisis",                        "category": "Geopolitical", "impact": "High"},
])
MARKET_EVENTS["date"] = pd.to_datetime(MARKET_EVENTS["date"])

FEATURE_COLS = ['Return', 'Price_vs_MA5', 'Price_vs_MA20',
                'Volume_zscore', 'Volatility_5d', 'PV_signal']

CAT_COLORS = {"Macro": "#4fc3f7", "Geopolitical": "#ef5350", "Earnings": "#66bb6a"}

# ── Data loading & feature engineering ────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(start="2020-01-01", end="2024-12-31"):
    raw    = yf.download(list(TICKERS.keys()), start=start, end=end,
                         auto_adjust=True, threads=True, progress=False)
    close  = raw["Close"].rename(columns=TICKERS)
    volume = raw["Volume"].rename(columns=TICKERS)
    close  = close.dropna(axis=1, thresh=int(len(close) * 0.8))
    volume = volume[close.columns]
    return close, volume

# (rest of file unchanged except KPI metrics block and validation rate formatting)