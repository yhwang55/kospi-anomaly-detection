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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KOSPI Anomaly Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Translations ---
TRANSLATIONS = {
    'en': {
        'tabs': [
            'Anomaly Detection', 'Feature Analysis', 'Cross-Stock Comparison', 'Market Event Validation',
            'Macro Overview', 'Regime Analysis', 'Factor Attribution'
        ],
        'dashboard_note': 'Use the tabs below to switch between anomaly detection, feature analysis, cross-stock comparison, market event validation, macro overview, regime analysis, and factor attribution.',
        'hero_kicker': 'Market Regime Intelligence',
        'hero_title': 'KOSPI anomaly signals, cleaned for fast decision-making.',
        'hero_subtitle': 'Explore abnormal trading days for {ticker} under the {date_range} window, compare how anomalies differ from normal sessions, and validate signals against curated macro, geopolitical, and earnings events.',
        'hero_pills': ['Model', 'Universe', 'Anomalies', 'Range', 'Analyst'],
            'spinner_loading_data': 'Loading market data...',
            'spinner_running_if': 'Running Isolation Forest...',
            'detect_title': 'Detection view',
            'detect_body': 'Filter the anomaly set by direction and return size, then inspect the price path, score curve, and the strongest feature driver for a selected date.',
            'anomaly_view': 'Anomaly view',
            'anomaly_view_options': ['All', 'Downside', 'Upside'],
            'min_return_label': 'Minimum |return| (%)',
            'top_rows': 'Top rows',
            'price_chart_header': 'Price Chart with Anomaly Flags',
            'top_anomaly_dates': 'Top anomaly dates',
            'why_flagged': 'Why this was flagged',
            'inspect_date': 'Inspect anomaly date',
            'no_anomalies_detected': 'No anomalies were detected in the selected period.',
            'features_title': 'Feature anatomy',
            'features_body': 'Compare the normal and anomalous distributions to see which inputs separate the model most clearly.',
            'features_distribution': 'Feature distribution: normal vs anomaly',
            'feature_separation': 'Feature separation power',
            'cross_stock_title': 'Cross-stock view',
            'cross_stock_body': 'Rank the KOSPI leaders by anomaly frequency and compare where stress concentrates across the universe.',
            'anomaly_rate_header': 'Anomaly rate across all 20 KOSPI stocks',
            'sort_summary_by': 'Sort summary by',
            'sort_options': ['Anomaly Rate (%)', 'Anomaly Days', 'Avg Anomaly Return'],
            'spinner_run_all': 'Running model on all stocks...',
            'detailed_summary_table': 'Detailed summary table',
            'event_validation_title': 'Event validation',
            'event_validation_body': 'Match the detected anomalies against curated market events and check whether the signal aligns with real shocks.',
            'validation_header': 'Anomaly detections vs known market events',
            'event_categories': 'Event categories',
            'show_only_validated': 'Show only validated anomalies',
            'kpi_anomalies_detected': 'Anomalies detected',
            'kpi_matched_events': 'Matched to events',
            'kpi_validation_rate': 'Validation rate',
            'no_market_events': 'No market events available.',
            'match_table_header': 'Anomaly ↔ event match table',
            'recent_curated_events': 'Recent curated market events',
            'macro_title': 'Macro overview',
            'macro_body': 'Recent macro, geopolitical, and policy events that commonly drive market anomalies.',
            'regime_title': 'Regime analysis',
            'regime_body': 'View regime diagnostics such as rolling volatility and anomaly concentration over time.',
            'rolling_volatility': '30-day rolling volatility',
            'vol_yaxis': '30d Vol (%)',
            'factor_title': 'Factor attribution',
            'factor_body': 'Aggregate which input features explain anomaly days versus normal days.',
            'feature_attribution_header': 'Feature attribution (separation)',
            'validated_by_category': 'Validated anomalies by event category',
        'sidebar_model_params': 'Model Parameters',
        'select_stock': 'Select stock',
        'forest_depth': 'Forest depth',
        'anomaly_sensitivity': 'Anomaly sensitivity (σ)',
        'analysis_window': 'Analysis window',
        'window': 'Window',
        'download_filtered': 'Download filtered anomalies',
        'download_stock': 'Download stock comparison',
        'download_validation': 'Download validation table',
        'no_anomalies_info': 'No anomalies match the current filters.',
        'not_enough_data': 'Not enough data to compute factor attribution for this ticker/window.'
    },
    'ko': {
        'tabs': ['이상치 탐지', '특성 분석', '종목 비교', '이벤트 검증', '거시 개요', '레짐 분석', '요인 기여'],
        'dashboard_note': '아래 탭에서 이상치 탐지, 특성 분석, 종목 비교, 이벤트 검증, 거시 개요, 레짐 분석, 요인 기여를 전환하세요.',
        'hero_kicker': '시장 레짐 인사이트',
        'hero_title': 'KOSPI 이상 신호',
        'hero_subtitle': '{ticker}의 {date_range} 기간 동안 이상 거래일을 탐색하고 정상일과 비교하며 거시·정책·실적 이벤트와 정합성을 검증합니다.',
        'hero_pills': ['모델', '유니버스', '이상치', '기간', '분석가'],
        'spinner_loading_data': '시장 데이터 로딩...',
        'spinner_running_if': '모델 실행 중...',
        'detect_title': '탐지',
        'detect_body': '방향과 수익 크기로 이상치를 필터링하고 선택한 날짜의 가격·점수·주요 특성 드라이버를 확인하세요.',
        'anomaly_view': '이상치 보기',
        'anomaly_view_options': ['전체', '하락', '상승'],
        'min_return_label': '최소 절대수익률 (%)',
        'top_rows': '상위 개수',
        'price_chart_header': '이상치 표시 가격 차트',
        'top_anomaly_dates': '주요 이상치 날짜',
        'why_flagged': '탐지 이유',
        'inspect_date': '검사할 날짜',
        'no_anomalies_detected': '해당 기간에 이상치가 없습니다.',
        'features_title': '특성 분석',
        'features_body': '정상일과 이상일의 분포를 비교해 모델을 잘 구분하는 특성을 확인합니다.',
        'features_distribution': '특성 분포: 정상 vs 이상',
        'feature_separation': '특성 분리력',
        'cross_stock_title': '종목 비교',
        'cross_stock_body': 'KOSPI 종목을 이상 빈도로 정렬해 스트레스 집중도를 비교합니다.',
        'anomaly_rate_header': 'KOSPI 20개 종목의 이상 비율',
        'sort_summary_by': '정렬 기준',
        'sort_options': ['이상치 비율 (%)', '이상치 일수', '평균 이상치 수익률'],
        'spinner_run_all': '모든 종목 모델 실행...',
        'detailed_summary_table': '상세 요약',
        'event_validation_title': '이벤트 검증',
        'event_validation_body': '감지된 이상치를 관리된 이벤트와 비교해 신호의 정합성을 확인합니다.',
        'validation_header': '이상치와 알려진 이벤트 비교',
        'event_categories': '이벤트 카테고리',
        'show_only_validated': '검증된 이상치만 보기',
        'kpi_anomalies_detected': '감지된 이상치',
        'kpi_matched_events': '매칭된 이벤트 수',
        'kpi_validation_rate': '검증 비율',
        'no_market_events': '시장 이벤트가 없습니다.',
        'match_table_header': '이상치 ↔ 이벤트 매칭',
        'recent_curated_events': '최근 이벤트',
        'macro_title': '거시 개요',
        'macro_body': '시장에 영향을 준 주요 거시·지정학·정책 이벤트 요약입니다.',
        'regime_title': '레짐 분석',
        'regime_body': '롤링 변동성 및 이상치 집중도 등 레짐 지표를 확인하세요.',
        'rolling_volatility': '30일 롤링 변동성',
        'vol_yaxis': '30일 변동성 (%)',
        'factor_title': '요인 기여',
        'factor_body': '어떤 특성이 이상치를 설명하는지 집계합니다.',
        'feature_attribution_header': '특성 기여 (분리)',
        'validated_by_category': '카테고리별 검증된 이상치',
        'sidebar_model_params': '모델 파라미터',
        'select_stock': '종목 선택',
        'forest_depth': '트리 개수',
        'anomaly_sensitivity': '민감도 (σ)',
        'analysis_window': '분석 기간',
        'window': '기간',
        'download_filtered': '필터된 이상치 다운로드',
        'download_stock': '종목 비교 다운로드',
        'download_validation': '검증 테이블 다운로드',
        'no_anomalies_info': '필터에 맞는 이상치가 없습니다.',
        'not_enough_data': '요인 기여를 계산할 충분한 데이터가 없습니다.'
    }
}

def tr(key):
    lang = st.session_state.get('ui_lang', 'en')
    parts = key.split('.')
    node = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    for p in parts:
        node = node.get(p, None) if isinstance(node, dict) else None
        if node is None:
            return TRANSLATIONS['en'].get(key, key)
    return node

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg: #0b1020;
        --panel: rgba(15, 21, 39, 0.78);
        --panel-strong: #131a2f;
        --line: rgba(148, 163, 184, 0.16);
        --text: #e8edf7;
        --muted: #93a1ba;
        --cyan: #66d9ff;
        --red: #ff6b6b;
        --green: #6ee7b7;
        --gold: #ffd166;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(102, 217, 255, 0.16), transparent 34%),
            radial-gradient(circle at top right, rgba(255, 107, 107, 0.10), transparent 28%),
            linear-gradient(180deg, #080d18 0%, var(--bg) 100%);
        color: var(--text);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.6rem;
        padding-bottom: 2.2rem;
    }

    h1, h2, h3, .section-header {
        font-family: 'Space Grotesk', sans-serif;
    }

    .metric-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(22, 30, 52, 0.92), rgba(14, 20, 35, 0.96));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 18px 16px;
        text-align: center;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(16px);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, rgba(102, 217, 255, 0.9), rgba(110, 231, 183, 0.85), rgba(255, 209, 102, 0.85));
    }
    .metric-label { color: var(--muted); font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-value { color: var(--text); font-size: 28px; font-weight: 700; margin-top: 8px; }
    .metric-delta { font-size: 12px; margin-top: 6px; color: var(--muted); }
    .section-header {
        color: #dbe6ff;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--line);
    }
    .compact-panel {
        background: rgba(15, 20, 35, 0.72);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: var(--muted);
        backdrop-filter: blur(14px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(19, 27, 48, 0.94), rgba(10, 15, 28, 0.96));
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 24px 26px;
        margin: 0 0 18px 0;
        box-shadow: 0 22px 48px rgba(0, 0, 0, 0.34);
        backdrop-filter: blur(18px);
    }
    .hero-kicker {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        color: var(--text);
        font-size: 34px;
        font-weight: 700;
        line-height: 1.08;
        margin: 0;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
        margin-top: 10px;
        max-width: 980px;
    }
    .hero-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 7px 12px;
        color: #d9e2f2;
        font-size: 12px;
        font-weight: 600;
    }
    .hero-pill strong {
        color: var(--cyan);
        font-weight: 700;
    }
    .dashboard-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 6px;
    }
    .tab-card {
        background: rgba(15, 20, 35, 0.72);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 14px;
        backdrop-filter: blur(14px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }
    .tab-card:hover {
        transform: translateY(-1px);
        border-color: rgba(102, 217, 255, 0.24);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
    }
    .tab-card-head {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .tab-card-title {
        color: #eef4ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0;
    }
    .tab-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 8px;
        background: rgba(102, 217, 255, 0.10);
        color: var(--cyan);
        flex: 0 0 auto;
        border: 1px solid rgba(102, 217, 255, 0.12);
    }
    .tab-card-body {
        color: var(--muted);
        font-size: 12.8px;
        line-height: 1.55;
    }
    .section-shell {
        background: rgba(12, 17, 31, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 24px;
        padding: 18px;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.2);
        margin-bottom: 16px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(14, 19, 35, 0.98), rgba(8, 12, 22, 0.98));
        border-right: 1px solid var(--line);
    }
    div[data-testid="stSidebar"] * { color: #d8e1f2; }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        letter-spacing: 0.01em;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.92rem;
        font-weight: 700;
        color: #dbe6ff;
        margin: 0.25rem 0 0.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background: rgba(255, 255, 255, 0.045);
        border-radius: 14px;
        border-color: rgba(148, 163, 184, 0.12);
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] svg {
        fill: #cbd5e1;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding-top: 0.05rem;
        padding-bottom: 0.12rem;
    }

    [data-testid="stPlotlyChart"] {
        animation: chartFadeIn 520ms ease-out both;
        transition: transform 220ms ease, opacity 220ms ease;
    }

    @keyframes chartFadeIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: linear-gradient(135deg, var(--cyan), var(--green));
        box-shadow: 0 0 0 6px rgba(102, 217, 255, 0.08);
    }

    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #8ea0bf;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #dbe6ff;
    }

    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stMultiSelect > label,
    [data-testid="stSidebar"] .stCheckbox > label {
        margin-bottom: 0.24rem;
    }

    [data-testid="stSidebar"] .sidebar-meta {
        color: #aeb8cf;
        font-size: 0.8rem;
        line-height: 1.45;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.10);
        padding: 10px 12px;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.10);
        margin: 0.7rem 0;
    }

    .stSelectbox, .stSlider, .stMultiSelect, .stCheckbox {
        margin-bottom: 8px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(102, 217, 255, 0.24);
        background: linear-gradient(135deg, rgba(102, 217, 255, 0.16), rgba(110, 231, 183, 0.12));
        color: #eef5ff;
        font-weight: 600;
        transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(102, 217, 255, 0.5);
        background: linear-gradient(135deg, rgba(102, 217, 255, 0.22), rgba(110, 231, 183, 0.16));
    }

    div[data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 20, 35, 0.45);
        padding: 8px;
        border: 1px solid var(--line);
        border-radius: 16px;
    }
    button[data-baseweb="tab"] {
        border-radius: 12px;
        background: transparent;
        color: var(--muted);
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(102, 217, 255, 0.14);
        color: #ffffff;
    }

    .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
    }

    .stMetric {
        background: transparent;
    }
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

DATE_RANGES = {
    "2020–2021": ("2020-01-01", "2021-12-31"),
    "2020–2022": ("2020-01-01", "2022-12-31"),
    "2020–2023": ("2020-01-01", "2023-12-31"),
    "2020–2024": ("2020-01-01", "2024-12-31"),
}

CHART_COLORS = {
    "price": "#7dd3fc",
    "anomaly": "#ff6b6b",
    "normal": "#5b6b8c",
    "volume": "#9aa8c7",
    "score": "#c4b5fd",
    "teal": "#6ee7b7",
    "amber": "#fbbf24",
    "ink": "#10172a",
}


def apply_premium_chart_style(fig, height, legend_y=1.02):
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#e8edf7'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=legend_y,
            xanchor='right',
            x=1,
            font=dict(size=11, color='#dbe6ff'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=0, r=0, t=14, b=0),
        hoverlabel=dict(
            bgcolor='rgba(12, 17, 31, 0.96)',
            bordercolor='rgba(148, 163, 184, 0.18)',
            font=dict(color='#f8fbff', family='IBM Plex Sans'),
        ),
    )
    fig.update_xaxes(
        gridcolor='rgba(148, 163, 184, 0.10)',
        zeroline=False,
        showline=False,
        tickfont=dict(color='#93a1ba'),
    )
    fig.update_yaxes(
        gridcolor='rgba(148, 163, 184, 0.10)',
        zeroline=False,
        showline=False,
        tickfont=dict(color='#93a1ba'),
    )
    return fig

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


def build_features(close_df, volume_df, ticker):
    df = pd.DataFrame()
    df['Close']  = close_df[ticker]
    df['Volume'] = volume_df[ticker]
    df['Return']        = df['Close'].pct_change()
    df['MA5']           = df['Close'].rolling(5).mean()
    df['MA20']          = df['Close'].rolling(20).mean()
    df['Price_vs_MA5']  = (df['Close'] - df['MA5'])  / df['MA5']
    df['Price_vs_MA20'] = (df['Close'] - df['MA20']) / df['MA20']
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std  = df['Volume'].rolling(20).std()
    df['Volume_zscore'] = (df['Volume'] - vol_mean) / vol_std
    df['Volatility_5d'] = df['Return'].rolling(5).std()
    df['PV_signal']     = df['Return'] * df['Volume_zscore']
    return df.dropna()


def explain_anomaly(row, baseline_df):
    baseline_mean = baseline_df[FEATURE_COLS].mean()
    baseline_std = baseline_df[FEATURE_COLS].std().replace(0, np.nan)
    z_scores = ((row[FEATURE_COLS] - baseline_mean) / baseline_std).fillna(0)
    impact = z_scores.abs().sort_values(ascending=False)
    return z_scores, impact


def df_to_csv_bytes(df):
    return df.to_csv(index=True).encode("utf-8-sig")


def svg_icon(kind):
    icons = {
        "detect": """
            <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path d='M7 1.5V3.2' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <path d='M7 10.8V12.5' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <path d='M1.5 7H3.2' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <path d='M10.8 7H12.5' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <circle cx='7' cy='7' r='3.1' stroke='currentColor' stroke-width='1.2'/>
            </svg>
        """,
        "features": """
            <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path d='M2 11.2V2.8' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <path d='M2 11.2H12' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <path d='M4 8.8L6.3 6.5L8 7.9L10.7 4.8' stroke='currentColor' stroke-width='1.2' stroke-linecap='round' stroke-linejoin='round'/>
            </svg>
        """,
        "stocks": """
            <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <rect x='2' y='7.3' width='1.8' height='4.2' rx='0.5' fill='currentColor'/>
                <rect x='6.1' y='4.8' width='1.8' height='6.7' rx='0.5' fill='currentColor'/>
                <rect x='10.2' y='2.8' width='1.8' height='8.7' rx='0.5' fill='currentColor'/>
            </svg>
        """,
        "event": """
            <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path d='M4 2.5V4' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <path d='M10 2.5V4' stroke='currentColor' stroke-width='1.2' stroke-linecap='round'/>
                <rect x='2.2' y='3.5' width='9.6' height='8.3' rx='1.4' stroke='currentColor' stroke-width='1.2'/>
                <path d='M3.5 6.2H10.5' stroke='currentColor' stroke-width='1.1' stroke-linecap='round'/>
                <path d='M3.8 8.4H8.8' stroke='currentColor' stroke-width='1.1' stroke-linecap='round'/>
            </svg>
        """,
            "macro": """
                <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M7 1.5C9.485 1.5 11.5 3.515 11.5 6s-2.015 4.5-4.5 4.5S2.5 8.485 2.5 6 4.515 1.5 7 1.5z' stroke='currentColor' stroke-width='1.0' stroke-linecap='round' stroke-linejoin='round'/>
                    <path d='M1.8 10.8c1.2-.9 3.2-1.6 5.2-1.6s4 .7 5.2 1.6' stroke='currentColor' stroke-width='1.0' stroke-linecap='round' stroke-linejoin='round'/>
                </svg>
            """,
            "regime": """
                <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M2 11h2.2l2.2-3 2.2 2 2.2-4 1.2 2' stroke='currentColor' stroke-width='1.2' stroke-linecap='round' stroke-linejoin='round'/>
                    <path d='M2 13h10' stroke='currentColor' stroke-width='1.0' stroke-linecap='round'/>
                </svg>
            """,
            "factor": """
                <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
                    <circle cx='4' cy='4' r='1.3' fill='currentColor'/>
                    <circle cx='10' cy='4' r='1.3' fill='currentColor'/>
                    <circle cx='7' cy='10' r='1.3' fill='currentColor'/>
                    <path d='M4.9 4.5L6.8 9.1' stroke='currentColor' stroke-width='0.9' stroke-linecap='round'/>
                    <path d='M9.1 4.5L7.2 9.1' stroke='currentColor' stroke-width='0.9' stroke-linecap='round'/>
                </svg>
            """,
    }
    return icons[kind]


@st.cache_data(show_spinner=False)
def load_market_events():
    events_path = Path(__file__).resolve().parent / "market_events.csv"
    events = pd.read_csv(events_path, parse_dates=['date'])
    return events.sort_values('date').reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_model(ticker, n_estimators, threshold_sigma, start_date, end_date):
    close, volume = load_data(start_date, end_date)
    df = build_features(close, volume, ticker)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    model    = IsolationForest(n_estimators=n_estimators,
                               contamination='auto', random_state=42)
    scores   = model.fit(X_scaled).score_samples(X_scaled)
    threshold = scores.mean() - threshold_sigma * scores.std()
    df['Score']      = scores
    df['Is_Anomaly'] = scores < threshold
    return df


@st.cache_data(show_spinner=False)
def run_all_stocks(n_estimators, threshold_sigma, start_date, end_date):
    close, volume = load_data(start_date, end_date)
    results = {}
    scaler = StandardScaler()
    for ticker in close.columns:
        try:
            df  = build_features(close, volume, ticker)
            X   = scaler.fit_transform(df[FEATURE_COLS])
            mdl = IsolationForest(n_estimators=n_estimators,
                                  contamination='auto', random_state=42)
            sc  = mdl.fit(X).score_samples(X)
            thr = sc.mean() - threshold_sigma * sc.std()
            pred = sc < thr
            results[ticker] = {
                'Anomaly Rate (%)':   round(pred.mean() * 100, 2),
                'Anomaly Days':       int(pred.sum()),
                'Avg Anomaly Return': round(df[pred]['Return'].mean() * 100, 3),
                'Score Threshold':    round(thr, 4),
            }
        except Exception:
            pass
    return pd.DataFrame(results).T

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    lang_map = {'English':'en', '한국어':'ko'}
    cur_lang = st.session_state.get('ui_lang', 'ko')
    cur_display = [k for k,v in lang_map.items() if v == cur_lang][0]
    sel = st.selectbox("Language / 언어", options=list(lang_map.keys()), index=list(lang_map.keys()).index(cur_display))
    st.session_state['ui_lang'] = lang_map[sel]

    st.markdown(f"## {tr('sidebar_model_params')}")
    ticker_name = st.selectbox(tr('select_stock'), sorted(TICKERS.values()), index=0)
    ticker_code = {v: k for k, v in TICKERS.items()}[ticker_name]

    st.markdown("---")
    n_estimators = st.slider(tr('forest_depth'), 50, 500, 200, 50)
    threshold_sigma = st.slider(tr('anomaly_sensitivity'), 1.0, 3.0, 2.0, 0.1,
                                help="Lower values surface more anomalies")

    st.markdown("---")
    st.markdown(f"### {tr('analysis_window')}")
    date_range = st.select_slider(
        tr('window'),
        options=list(DATE_RANGES.keys()),
        value="2020–2024"
    )
    start_date, end_date = DATE_RANGES[date_range]

    st.markdown("---")
    st.markdown("""
    <div style='color:#aeb8cf; font-size:12px; line-height:1.65; background:rgba(255,255,255,0.03); border:1px solid rgba(148,163,184,0.12); padding:12px 14px; border-radius:14px'>
    <b style='color:#e8edf7'>Model:</b> Isolation Forest<br>
    <b style='color:#e8edf7'>Universe:</b> KOSPI Top-20 by market cap<br>
    <b style='color:#e8edf7'>Features:</b> Return, MA deviation, Volume Z-score, Volatility, PV signal<br>
    <b style='color:#e8edf7'>Data:</b> yfinance · 2020–2024
    </div>
    """, unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
MARKET_EVENTS = load_market_events()

with st.spinner(tr('spinner_loading_data')):
    close, volume = load_data(start_date, end_date)

with st.spinner(tr('spinner_running_if')):
    df_feat = run_model(ticker_name, n_estimators, threshold_sigma, start_date, end_date)

anomalies  = df_feat[df_feat['Is_Anomaly']]
n_anom     = len(anomalies)
total_days = len(df_feat)
anom_rate  = n_anom / total_days * 100
avg_ret    = anomalies['Return'].mean() * 100
norm_ret   = df_feat[~df_feat['Is_Anomaly']]['Return'].mean() * 100
ret_mult   = abs(avg_ret / norm_ret) if norm_ret != 0 else 0
hero_anomaly_rate = f"{anom_rate:.1f}%"

# ── Header ─────────────────────────────────────────────────────────────────
hero_sub = tr('hero_subtitle').format(ticker=ticker_name, date_range=date_range)
hero_pills = tr('hero_pills')
st.markdown(f"""
<div class='hero-card'>
    <div class='hero-kicker'>{tr('hero_kicker')}</div>
    <h1 class='hero-title'>{tr('hero_title')}</h1>
    <div class='hero-subtitle'>{hero_sub}</div>
    <div class='hero-pill-row'>
        <div class='hero-pill'><strong>{hero_pills[0]}</strong> Isolation Forest</div>
        <div class='hero-pill'><strong>{hero_pills[1]}</strong> KOSPI Top-20</div>
        <div class='hero-pill'><strong>{hero_pills[2]}</strong> {hero_anomaly_rate}</div>
        <div class='hero-pill'><strong>{hero_pills[3]}</strong> {date_range}</div>
        <div class='hero-pill'><strong>{hero_pills[4]}</strong> Yoon Hwang</div>
    </div>
    <div class='dashboard-note'>
        {tr('dashboard_note')}
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-shell'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Trading Days Analyzed", f"{total_days:,}", None),
    (c2, "Anomaly Days Detected",  f"{n_anom}",       f"{anom_rate:.1f}% of total"),
    (c3, "Avg Anomaly Return",     f"{avg_ret:+.2f}%", f"vs {norm_ret:+.2f}% normal"),
    (c4, "Return Magnitude",       f"{ret_mult:.1f}×", "anomaly vs normal"),
    (c5, "Model Trees",            f"{n_estimators}",  f"σ threshold: {threshold_sigma}"),
]
for col, label, val, delta in metrics:
    delta_html = f"<div class='metric-delta' style='color:#8b95b0'>{delta}</div>" if delta else ""
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{val}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"<div class='dashboard-note'>{tr('dashboard_note')}</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Tab layout ─────────────────────────────────────────────────────────────
tab_names = tr('tabs')
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('detect')}</span>
                <div class='tab-card-title'>{tr('detect_title')}</div>
            </div>
            <div class='tab-card-body'>
                {tr('detect_body')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    anomaly_view, min_abs_return, top_n = st.columns([1.2, 1.2, 1])
    with anomaly_view:
        anomaly_filter_display = st.selectbox(
            tr('anomaly_view'),
            options=tr('anomaly_view_options'),
            index=0,
        )
        # map translated display back to canonical filter values
        CANONICAL_ANOMALY_OPTIONS = ['All', 'Downside', 'Upside']
        try:
            idx_af = tr('anomaly_view_options').index(anomaly_filter_display)
        except ValueError:
            idx_af = 0
        anomaly_filter = CANONICAL_ANOMALY_OPTIONS[idx_af]
    with min_abs_return:
        min_return_pct = st.slider(
            tr('min_return_label'),
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )
    with top_n:
        top_n_value = st.slider(tr('top_rows'), 5, 15, 10, 1)

    filtered_anomalies = anomalies.copy()
    if anomaly_filter == "Downside":
        filtered_anomalies = filtered_anomalies[filtered_anomalies['Return'] < 0]
    elif anomaly_filter == "Upside":
        filtered_anomalies = filtered_anomalies[filtered_anomalies['Return'] > 0]
    if min_return_pct > 0:
        filtered_anomalies = filtered_anomalies[filtered_anomalies['Return'].abs() * 100 >= min_return_pct]
    filtered_anomalies = filtered_anomalies.sort_values('Score')

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown(f"<div class='section-header'>{tr('price_chart_header')}</div>",
                unsafe_allow_html=True)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            vertical_spacing=0.04)

        # Price + anomaly scatter
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat['Close'],
            mode='lines', name='Close Price',
            line=dict(color=CHART_COLORS['price'], width=1.8),
        ), row=1, col=1)

        normal_df = df_feat[~df_feat['Is_Anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_df.index, y=normal_df['Close'],
            mode='markers', name='Normal',
            marker=dict(color=CHART_COLORS['normal'], size=3, opacity=0.22),
            showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=filtered_anomalies.index, y=filtered_anomalies['Close'],
            mode='markers', name='Anomaly',
            marker=dict(color=CHART_COLORS['anomaly'], size=10, symbol='circle',
                        line=dict(color='#ffd1d1', width=1.5)),
        ), row=1, col=1)

        # Volume bars
        colors_vol = ['#ef5350' if a else '#37474f'
                      for a in df_feat['Is_Anomaly']]
        fig.add_trace(go.Bar(
            x=df_feat.index, y=df_feat['Volume'],
            name='Volume', marker_color=colors_vol, opacity=0.8,
        ), row=2, col=1)

        # Anomaly score
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat['Score'],
            mode='lines', name='Anomaly Score',
            line=dict(color=CHART_COLORS['score'], width=1.4),
            fill='tozeroy', fillcolor='rgba(196,181,253,0.10)',
        ), row=3, col=1)

        thr_val = df_feat['Score'].mean() - threshold_sigma * df_feat['Score'].std()
        fig.add_hline(y=thr_val, line_dash='dash',
                      line_color=CHART_COLORS['anomaly'], line_width=1,
                      annotation_text=f'Threshold (μ−{threshold_sigma}σ)',
                      annotation_font_color=CHART_COLORS['anomaly'],
                      row=3, col=1)

        apply_premium_chart_style(fig, height=540, legend_y=1.01)
        fig.update_layout(transition=dict(duration=520, easing='cubic-in-out'))
        fig.update_yaxes(gridcolor='#1e2130', row=1, col=1, title_text='Price (KRW)')
        fig.update_yaxes(gridcolor='#1e2130', row=2, col=1, title_text='Volume')
        fig.update_yaxes(gridcolor='#1e2130', row=3, col=1, title_text='IF Score')
        fig.update_xaxes(gridcolor='#1e2130')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown(f"<div class='section-header'>{tr('top_anomaly_dates')}</div>",
                unsafe_allow_html=True)
        st.download_button(
            tr('download_filtered'),
            data=df_to_csv_bytes(filtered_anomalies.head(top_n_value)[['Close', 'Volume', 'Return', 'Volume_zscore', 'Score', 'Is_Anomaly']]),
            file_name=f"{ticker_name.replace(' ', '_').lower()}_anomalies.csv",
            mime="text/csv",
            use_container_width=True,
        )

        top10 = (filtered_anomalies[['Close', 'Return', 'Volume_zscore', 'Score']]
                 .head(top_n_value).copy())
        top10['Return']       = (top10['Return'] * 100).round(2)
        top10['Volume_zscore'] = top10['Volume_zscore'].round(2)
        top10['Score']        = top10['Score'].round(4)
        top10.index = top10.index.strftime('%Y-%m-%d')
        top10.columns = ['Price', 'Ret%', 'Vol-Z', 'Score']

        if top10.empty:
            st.info(tr('no_anomalies_info'))

        for date, row in top10.iterrows():
            color = '#ef5350' if row['Ret%'] < 0 else '#66bb6a'
            st.markdown(f"""
            <div style='background:#1e2130; border-left:3px solid {color};
                        padding:8px 12px; border-radius:6px; margin-bottom:6px'>
                <div style='color:#c5cae9; font-size:12px; font-weight:600'>{date}</div>
                <div style='color:{color}; font-size:14px; font-weight:700'>{row['Ret%']:+.2f}%</div>
                <div style='color:#8b95b0; font-size:11px'>Vol-Z: {row['Vol-Z']:.2f} · Score: {row['Score']:.4f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='section-header'>{tr('why_flagged')}</div>",
                unsafe_allow_html=True)
        if len(anomalies) > 0:
            selected_date = st.selectbox(
                tr('inspect_date'),
                options=anomalies.index.strftime('%Y-%m-%d').tolist(),
                index=0,
                label_visibility='collapsed'
            )
            selected_dt = pd.Timestamp(selected_date)
            selected_row = df_feat.loc[selected_dt]
            z_scores, impact = explain_anomaly(selected_row, df_feat[~df_feat['Is_Anomaly']])

            driver_df = pd.DataFrame({
                'Feature': impact.head(4).index,
                'Z-Score': z_scores[impact.head(4).index].values,
            })

            fig_explain = go.Figure(go.Bar(
                x=driver_df['Z-Score'],
                y=driver_df['Feature'],
                orientation='h',
                marker_color=[CHART_COLORS['anomaly'] if v < 0 else CHART_COLORS['price'] for v in driver_df['Z-Score']],
                text=[f"{v:+.2f}σ" for v in driver_df['Z-Score']],
                textposition='outside',
            ))
            apply_premium_chart_style(fig_explain, height=240, legend_y=1.05)
            fig_explain.update_layout(margin=dict(l=0, r=10, t=0, b=0), xaxis_title='Deviation vs normal baseline', yaxis_title='', showlegend=False)
            fig_explain.update_xaxes(gridcolor='#1e2130')
            st.plotly_chart(fig_explain, use_container_width=True)

            top_driver = driver_df.iloc[0]
            st.caption(
                f"Largest deviation: {top_driver['Feature']} ({top_driver['Z-Score']:+.2f}σ vs normal days)"
            )
        else:
            st.info("No anomalies were detected in the selected period.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Feature Analysis
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('features')}</span>
                <div class='tab-card-title'>{tr('features_title')}</div>
            </div>
            <div class='tab-card-body'>
                {tr('features_body')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-header'>{tr('features_distribution')}</div>",
                unsafe_allow_html=True)

    normal_df  = df_feat[~df_feat['Is_Anomaly']]
    anomaly_df = df_feat[df_feat['Is_Anomaly']]

    fig2 = make_subplots(rows=2, cols=3,
                         subplot_titles=FEATURE_COLS,
                         vertical_spacing=0.15, horizontal_spacing=0.08)

    for idx, col in enumerate(FEATURE_COLS):
        r, c = divmod(idx, 3)
        fig2.add_trace(go.Histogram(
            x=normal_df[col].dropna(), name='Normal',
            marker_color=CHART_COLORS['price'], opacity=0.58,
            nbinsx=40, histnorm='probability density',
            showlegend=(idx == 0),
        ), row=r+1, col=c+1)
        fig2.add_trace(go.Histogram(
            x=anomaly_df[col].dropna(), name='Anomaly',
            marker_color=CHART_COLORS['anomaly'], opacity=0.82,
            nbinsx=20, histnorm='probability density',
            showlegend=(idx == 0),
        ), row=r+1, col=c+1)

    fig2.update_layout(
        height=480, template='plotly_dark', barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.03,
                    xanchor='right', x=1, font=dict(color='#dbe6ff')),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig2.update_xaxes(gridcolor='#1e2130')
    fig2.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig2, use_container_width=True)

    # Feature importance (mean absolute difference)
    st.markdown(f"<div class='section-header'>{tr('feature_separation')}</div>",
                unsafe_allow_html=True)
    sep = {}
    for col in FEATURE_COLS:
        n_mean = normal_df[col].mean()
        a_mean = anomaly_df[col].mean()
        n_std  = normal_df[col].std()
        sep[col] = abs(a_mean - n_mean) / (n_std + 1e-9)

    sep_df = pd.Series(sep).sort_values(ascending=True)
    fig3 = go.Figure(go.Bar(
        x=sep_df.values, y=sep_df.index,
        orientation='h',
        marker_color=[CHART_COLORS['anomaly'] if v == sep_df.max() else CHART_COLORS['price']
                      for v in sep_df.values],
        text=[f'{v:.2f}σ' for v in sep_df.values],
        textposition='outside',
    ))
    apply_premium_chart_style(fig3, height=280, legend_y=1.05)
    fig3.update_layout(xaxis_title='Separation (Cohen''s d)', yaxis_title='', margin=dict(l=0, r=60, t=10, b=0), showlegend=False)
    fig3.update_xaxes(gridcolor='#1e2130')
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Cross-Stock Comparison
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('stocks')}</span>
                <div class='tab-card-title'>{tr('cross_stock_title')}</div>
            </div>
            <div class='tab-card-body'>
                {tr('cross_stock_body')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-header'>{tr('anomaly_rate_header')}</div>",
                unsafe_allow_html=True)

    sort_display = st.selectbox(
        tr('sort_summary_by'),
        options=tr('sort_options'),
        index=0,
    )

    with st.spinner(tr('spinner_run_all')):
        summary_df = run_all_stocks(n_estimators, threshold_sigma, start_date, end_date)

    # Map displayed (translated) sort option back to canonical column name
    CANONICAL_SORT_OPTIONS = ['Anomaly Rate (%)', 'Anomaly Days', 'Avg Anomaly Return']
    try:
        idx_sort = tr('sort_options').index(sort_display)
    except ValueError:
        idx_sort = 0
    sort_metric = CANONICAL_SORT_OPTIONS[idx_sort]
    ascending = sort_metric == 'Avg Anomaly Return'
    summary_df = summary_df.sort_values(sort_metric, ascending=ascending)
    rates = summary_df['Anomaly Rate (%)'].astype(float)

    fig4 = go.Figure(go.Bar(
        x=summary_df.index,
        y=rates,
        marker=dict(
            color=rates,
            colorscale=[[0.0, CHART_COLORS['teal']], [0.55, CHART_COLORS['price']], [1.0, CHART_COLORS['anomaly']]],
            showscale=True,
            colorbar=dict(title='Rate %', thickness=12),
        ),
        text=[f"{v:.1f}%" for v in rates],
        textposition='outside',
    ))
    avg = rates.mean()
    fig4.add_hline(y=avg, line_dash='dash', line_color=CHART_COLORS['price'],
                   annotation_text=f'Average {avg:.1f}%',
                   annotation_font_color=CHART_COLORS['price'])
    apply_premium_chart_style(fig4, height=390, legend_y=1.05)
    fig4.update_layout(yaxis_title='Anomaly Rate (%)', xaxis_title='', xaxis_tickangle=-35, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    fig4.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig4, use_container_width=True)

    # Summary table
    st.markdown(f"<div class='section-header'>{tr('detailed_summary_table')}</div>",
                unsafe_allow_html=True)
    display_df = summary_df.copy()
    display_df['Anomaly Rate (%)'] = display_df['Anomaly Rate (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Avg Anomaly Return'] = display_df['Avg Anomaly Return'].apply(lambda x: f"{x:+.3f}%")
    st.download_button(
        tr('download_stock'),
        data=df_to_csv_bytes(summary_df),
        file_name="kospi_stock_anomaly_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.dataframe(display_df, use_container_width=True, height=320)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: Market Event Validation
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('macro')}</span>
                <div class='tab-card-title'>{tr('event_validation_title')}</div>
            </div>
            <div class='tab-card-body'>
                {tr('event_validation_body')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-header'>{tr('validation_header')}</div>",
                unsafe_allow_html=True)

    event_categories = sorted(MARKET_EVENTS['category'].unique().tolist())
    selected_categories = st.multiselect(
        tr('event_categories'),
        options=event_categories,
        default=event_categories,
    )
    show_only_validated = st.checkbox(tr('show_only_validated'), value=False)

    WINDOW = pd.Timedelta(days=3)
    matches = []
    for dt in anomalies.index:
        nearby = MARKET_EVENTS[
            (MARKET_EVENTS['date'] >= dt - WINDOW) &
            (MARKET_EVENTS['date'] <= dt + WINDOW)
        ]
        if not nearby.empty:
            best = nearby.iloc[(nearby['date'] - dt).abs().argsort()[:1]]
            matches.append({
                'Anomaly Date': dt.strftime('%Y-%m-%d'),
                'Return (%)':   round(df_feat.loc[dt, 'Return'] * 100, 2),
                'Vol Z-score':  round(df_feat.loc[dt, 'Volume_zscore'], 2),
                'Event':        best['event'].values[0],
                'Category':     best['category'].values[0],
                'Impact':       best['impact'].values[0],
                '_validated':   True,
            })
        else:
            matches.append({
                'Anomaly Date': dt.strftime('%Y-%m-%d'),
                'Return (%)':   round(df_feat.loc[dt, 'Return'] * 100, 2),
                'Vol Z-score':  round(df_feat.loc[dt, 'Volume_zscore'], 2),
                'Event':        'No matching event found',
                'Category':     '—',
                'Impact':       '—',
                '_validated':   False,
            })

    match_df  = pd.DataFrame(matches)
    validated = match_df[match_df['_validated']]
    if selected_categories:
        match_df = match_df[(match_df['_validated'] == False) | (match_df['Category'].isin(selected_categories))]
        validated = match_df[match_df['_validated']]
    if show_only_validated:
        match_df = match_df[match_df['_validated']]
        validated = match_df[match_df['_validated']]
    val_rate  = len(validated) / len(match_df) * 100 if len(match_df) > 0 else 0

    # Validation KPIs
    k1, k2, k3 = st.columns(3)
    for col, label, val, color in [
        (k1, tr('kpi_anomalies_detected'),    str(len(match_df)),          CHART_COLORS['price']),
        (k2, tr('kpi_matched_events'),     str(len(validated)),          CHART_COLORS['teal']),
        (k3, tr('kpi_validation_rate'),       f"{val_rate:.0f}%",         CHART_COLORS['score']),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='color:{color}'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Annotated price chart
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['Close'],
        mode='lines', name='Close Price',
        line=dict(color=CHART_COLORS['price'], width=1.8),
    ))

    val_dates = pd.to_datetime(validated['Anomaly Date'])
    unval_dates = pd.to_datetime(match_df[~match_df['_validated']]['Anomaly Date'])

    confirmed_df   = df_feat[df_feat.index.isin(val_dates)]
    unconfirmed_df = df_feat[df_feat.index.isin(unval_dates)]

    fig5.add_trace(go.Scatter(
        x=confirmed_df.index, y=confirmed_df['Close'],
        mode='markers', name='Anomaly — Event Confirmed',
        marker=dict(color=CHART_COLORS['anomaly'], size=11, symbol='circle',
                    line=dict(color='#ffd1d1', width=1.5)),
    ))
    fig5.add_trace(go.Scatter(
        x=unconfirmed_df.index, y=unconfirmed_df['Close'],
        mode='markers', name='Anomaly — No Event Match',
        marker=dict(color=CHART_COLORS['amber'], size=8, symbol='circle', opacity=0.8),
    ))

    # Event annotations
    key_events = [
        ('2022-02-24', 'Russia-Ukraine'),
        ('2022-06-16', 'Fed +75bp'),
        ('2023-03-10', 'SVB'),
        ('2024-08-05', 'Carry Unwind'),
        ('2024-03-20', 'AI Chip Surge'),
        ('2024-12-04', 'Martial Law'),
    ]
    for date_str, label in key_events:
        dt = pd.Timestamp(date_str)
        if dt in df_feat.index:
            fig5.add_annotation(
                x=dt, y=df_feat.loc[dt, 'Close'],
                text=label, showarrow=True,
                arrowhead=2, arrowcolor='#93a1ba', arrowsize=0.8,
                font=dict(size=10, color='#e8edf7'),
                bgcolor='rgba(16,23,42,0.88)',
                bordercolor='rgba(148,163,184,0.22)', borderwidth=1,
                ay=-40,
            )

    apply_premium_chart_style(fig5, height=400, legend_y=1.01)
    fig5.update_layout(yaxis_title='Price (KRW)', margin=dict(l=0, r=0, t=10, b=0))
    fig5.update_xaxes(gridcolor='#1e2130')
    fig5.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig5, use_container_width=True)

    # Match table
    st.markdown(f"<div class='section-header'>{tr('match_table_header')}</div>",
                unsafe_allow_html=True)
    display_match = match_df.drop(columns=['_validated']).copy()
    st.download_button(
        tr('download_validation'),
        data=df_to_csv_bytes(display_match),
        file_name="kospi_event_validation.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.dataframe(display_match, use_container_width=True, height=300)


# ═════════════════════════════════════════════════════════════════════════=
# TAB 5: Macro Overview
# ═════════════════════════════════════════════════════════════════════════=
with tab5:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('event')}</span>
                <div class='tab-card-title'>Macro overview</div>
            </div>
            <div class='tab-card-body'>
                Recent macro, geopolitical, and policy events that commonly drive market anomalies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Recent curated market events</div>", unsafe_allow_html=True)
    if not MARKET_EVENTS.empty:
        recent = MARKET_EVENTS.sort_values('date', ascending=False).head(12).copy()
        recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(recent[['date', 'event', 'category', 'impact']], use_container_width=True, height=320)
    else:
        st.info(tr('no_market_events'))


# ═════════════════════════════════════════════════════════════════════════=
# TAB 6: Regime Analysis
# ═════════════════════════════════════════════════════════════════════════=
with tab6:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('regime')}</span>
                <div class='tab-card-title'>{tr('regime_title')}</div>
            </div>
            <div class='tab-card-body'>
                {tr('regime_body')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-header'>{tr('rolling_volatility')}</div>", unsafe_allow_html=True)
    vol30 = df_feat['Return'].rolling(30).std() * 100
    fig_reg = go.Figure(go.Scatter(x=vol30.index, y=vol30, mode='lines', line=dict(color=CHART_COLORS['teal'], width=1.6)))
    apply_premium_chart_style(fig_reg, height=380, legend_y=1.01)
    fig_reg.update_layout(yaxis_title=tr('vol_yaxis'), margin=dict(l=0, r=0, t=10, b=0))
    fig_reg.update_xaxes(gridcolor='#1e2130')
    fig_reg.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig_reg, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════=
# TAB 7: Factor Attribution
# ═════════════════════════════════════════════════════════════════════════=
with tab7:
    st.markdown(
        f"""
        <div class='tab-card'>
            <div class='tab-card-head'>
                <span class='tab-icon'>{svg_icon('factor')}</span>
                <div class='tab-card-title'>{tr('factor_title')}</div>
            </div>
            <div class='tab-card-body'>
                {tr('factor_body')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-header'>{tr('feature_attribution_header')}</div>", unsafe_allow_html=True)
    normal_df = df_feat[~df_feat['Is_Anomaly']]
    anomaly_df = df_feat[df_feat['Is_Anomaly']]
    if len(normal_df) > 5 and len(anomaly_df) > 0:
        sep = {}
        for col in FEATURE_COLS:
            n_mean = normal_df[col].mean()
            a_mean = anomaly_df[col].mean()
            n_std  = normal_df[col].std()
            sep[col] = abs(a_mean - n_mean) / (n_std + 1e-9)
        sep_df = pd.Series(sep).sort_values(ascending=True)
        fig_attr = go.Figure(go.Bar(x=sep_df.values, y=sep_df.index, orientation='h', marker_color=CHART_COLORS['price'], text=[f"{v:.2f}σ" for v in sep_df.values], textposition='outside'))
        apply_premium_chart_style(fig_attr, height=340, legend_y=1.01)
        fig_attr.update_layout(margin=dict(l=0, r=60, t=10, b=0), showlegend=False)
        fig_attr.update_xaxes(gridcolor='#1e2130')
        fig_attr.update_yaxes(gridcolor='#1e2130')
        st.plotly_chart(fig_attr, use_container_width=True)
    else:
        st.info(tr('not_enough_data'))

    # Category breakdown
    if len(validated) > 0:
        st.markdown(f"<div class='section-header'>{tr('validated_by_category')}</div>",
                unsafe_allow_html=True)
        cat_counts = validated['Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig6 = go.Figure(go.Pie(
            labels=cat_counts['Category'],
            values=cat_counts['Count'],
            marker_colors=[CAT_COLORS.get(c, '#8b95b0') for c in cat_counts['Category']],
            hole=0.5,
            textinfo='label+percent',
            textfont_size=13,
        ))
        fig6.update_layout(
            height=280, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='IBM Plex Sans', color='#e8edf7'),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig6, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5080; font-size:12px; padding:8px 0'>
    KOSPI Anomaly Detection · Yoon Hwang · UW–Madison Data Science & Economics · 2025<br>
    Isolation Forest · yfinance · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
