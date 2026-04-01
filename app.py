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
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KOSPI Anomaly Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
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


@st.cache_data(show_spinner=False)
def run_model(ticker, n_estimators, threshold_sigma):
    close, volume = load_data()
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
def run_all_stocks(n_estimators, threshold_sigma):
    close, volume = load_data()
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


@st.cache_data(show_spinner=False)
def load_tableau_data():
    df = pd.read_csv("tableau/kospi_dashboard_data.csv", parse_dates=["date"])
    return df.sort_values("date")

@st.cache_data(show_spinner=False)
def load_regression_summary():
    return pd.read_csv("results/regression_summary.csv")

@st.cache_data(show_spinner=False)
def load_shap_values():
    return pd.read_csv("results/shap_values.csv")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    ticker_name = st.selectbox("Stock", sorted(TICKERS.values()), index=0)
    ticker_code = {v: k for k, v in TICKERS.items()}[ticker_name]

    st.markdown("---")
    n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
    threshold_sigma = st.slider("Anomaly Threshold (σ below mean)", 1.0, 3.0, 2.0, 0.1,
                                help="Lower = more anomalies detected")

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    date_range = st.select_slider(
        "Analysis Period",
        options=["2020–2021", "2020–2022", "2020–2023", "2020–2024"],
        value="2020–2024"
    )
    end_year = date_range.split("–")[1]

    st.markdown("---")
    st.markdown("""
    <div style='color:#8b95b0; font-size:12px; line-height:1.6'>
    <b>Model:</b> Isolation Forest<br>
    <b>Universe:</b> KOSPI Top-20 by market cap<br>
    <b>Features:</b> Return, MA deviation,<br>
    &nbsp;&nbsp;Volume Z-score, Volatility, PV signal<br>
    <b>Data:</b> yfinance · 2020–2024
    </div>
    """, unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    close, volume = load_data()

with st.spinner("Running Isolation Forest..."): 
    df_feat = run_model(ticker_name, n_estimators, threshold_sigma)

anomalies  = df_feat[df_feat['Is_Anomaly']]
n_anom     = len(anomalies)
total_days = len(df_feat)
anom_rate  = n_anom / total_days * 100
avg_ret    = anomalies['Return'].mean() * 100
norm_ret   = df_feat[~df_feat['Is_Anomaly']]['Return'].mean() * 100
ret_mult   = abs(avg_ret / norm_ret) if norm_ret != 0 else 0

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='color:#e8ecf5; font-size:28px; font-weight:700; margin-bottom:4px'>
📈 KOSPI Anomaly Detection Dashboard
</h1>
<p style='color:#8b95b0; font-size:14px; margin-top:0'>
Isolation Forest · {ticker_name} · 2020–{end_year} &nbsp;|&nbsp;
Built by <b style='color:#c5cae9'>Yoon Hwang</b> · UW–Madison Data Science & Economics
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Trading Days Analyzed", f"{total_days:,}", None),
    (c2, "Anomaly Days Detected",  f"{n_anom}",       f"{anom_rate:.1f}% of total"),
    (c3, "Avg Anomaly Return",     f"{avg_ret:+.2f} %", f" vs {norm_ret:+.2f}% normal"),
    (c4, "Return Magnitude",       f"{ret_mult:.1f}×", "anomaly vs normal"),
    (c5, "Model Trees",            f"{n_estimators}",   f"σ threshold: {threshold_sigma}"),
]
for col, label, val, delta in metrics:
    delta_html = f"<div class='metric-delta' style='color:#8b95b0'>{delta}</div>" if delta else ""
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{val}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tab layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Anomaly Detection", "🔬 Feature Analysis",
    "🏢 Cross-Stock Comparison", "📅 Market Event Validation",
    "🌐 Macro Overview", "🧭 Regime Analysis", "🧮 Factor Attribution"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown("<div class='section-header'>Price Chart with Anomaly Flags</div>",
                    unsafe_allow_html=True)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            vertical_spacing=0.04)

        # Price + anomaly scatter
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat['Close'],
            mode='lines', name='Close Price',
            line=dict(color='#4fc3f7', width=1.5),
        ), row=1, col=1)

        normal_df = df_feat[~df_feat['Is_Anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_df.index, y=normal_df['Close'],
            mode='markers', name='Normal',
            marker=dict(color='#4fc3f7', size=3, opacity=0.3),
            showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies['Close'],
            mode='markers', name='Anomaly',
            marker=dict(color='#ef5350', size=9, symbol='circle',
                        line=dict(color='#ff8a80', width=1.5)),
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
            line=dict(color='#ce93d8', width=1),
            fill='tozeroy', fillcolor='rgba(206,147,216,0.08)',
        ), row=3, col=1)

        thr_val = df_feat['Score'].mean() - threshold_sigma * df_feat['Score'].std()
        fig.add_hline(y=thr_val, line_dash='dash',
                      line_color='#ef5350', line_width=1,
                      annotation_text=f'Threshold (μ−{threshold_sigma}σ)',
                      annotation_font_color='#ef5350',
                      row=3, col=1)

        fig.update_layout(
            height=560, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.01,
                        xanchor='right', x=1, font=dict(size=12)),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig.update_yaxes(gridcolor='#1e2130', row=1, col=1, title_text='Price (KRW)')
        fig.update_yaxes(gridcolor='#1e2130', row=2, col=1, title_text='Volume')
        fig.update_yaxes(gridcolor='#1e2130', row=3, col=1, title_text='IF Score')
        fig.update_xaxes(gridcolor='#1e2130')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Top Anomaly Dates</div>",
                    unsafe_allow_html=True)
        top10 = (anomalies[['Close', 'Return', 'Volume_zscore', 'Score']]
                 .sort_values('Score').head(10).copy())
        top10['Return']       = (top10['Return'] * 100).round(2)
        top10['Volume_zscore'] = top10['Volume_zscore'].round(2)
        top10['Score']        = top10['Score'].round(4)
        top10.index = top10.index.strftime('%Y-%m-%d')
        top10.columns = ['Price', 'Ret%', 'Vol-Z', 'Score']

        for date, row in top10.iterrows():
            color = '#ef5350' if row['Ret%'] < 0 else '#66bb6a'
            st.markdown(f"""
            <div style='background:#1e2130; border-left:3px solid {color};
                        padding:8px 12px; border-radius:6px; margin-bottom:6px'>
                <div style='color:#c5cae9; font-size:12px; font-weight:600'>{date}</div>
                <div style='color:{color}; font-size:14px; font-weight:700'>{row['Ret%']:+.2f}%</div>
                <div style='color:#8b95b0; font-size:11px'>Vol-Z: {row['Vol-Z']:.2f} · Score: {row['Score']:.4f}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Feature Analysis
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Feature Distribution: Normal vs Anomaly</div>",
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
            marker_color='#4fc3f7', opacity=0.6,
            nbinsx=40, histnorm='probability density',
            showlegend=(idx == 0),
        ), row=r+1, col=c+1)
        fig2.add_trace(go.Histogram(
            x=anomaly_df[col].dropna(), name='Anomaly',
            marker_color='#ef5350', opacity=0.8,
            nbinsx=20, histnorm='probability density',
            showlegend=(idx == 0),
        ), row=r+1, col=c+1)

    fig2.update_layout(
        height=480, template='plotly_dark', barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.03,
                    xanchor='right', x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig2.update_xaxes(gridcolor='#1e2130')
    fig2.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig2, use_container_width=True)

    # Feature importance (mean absolute difference)
    st.markdown("<div class='section-header'>Feature Separation Power</div>",
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
        marker_color=['#ef5350' if v == sep_df.max() else '#4fc3f7'
                      for v in sep_df.values],
        text=[f'{v:.2f}σ' for v in sep_df.values],
        textposition='outside',
    ))
    fig3.update_layout(
        height=280, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Separation (Cohen''s d)', yaxis_title='',
        margin=dict(l=0, r=60, t=10, b=0),
    )
    fig3.update_xaxes(gridcolor='#1e2130')
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Cross-Stock Comparison
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Anomaly Rate Across All 20 KOSPI Stocks</div>",
                unsafe_allow_html=True)

    with st.spinner("Running model on all stocks..."): 
        summary_df = run_all_stocks(n_estimators, threshold_sigma)

    summary_df = summary_df.sort_values('Anomaly Rate (%)', ascending=False)
    rates = summary_df['Anomaly Rate (%)'].astype(float)

    fig4 = go.Figure(go.Bar(
        x=summary_df.index,
        y=rates,
        marker=dict(
            color=rates,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Rate %', thickness=12),
        ),
        text=[f"{v:.1f}%" for v in rates],
        textposition='outside',
    ))
    avg = rates.mean()
    fig4.add_hline(y=avg, line_dash='dash', line_color='#4fc3f7',
                   annotation_text=f'Average {avg:.1f}%',
                   annotation_font_color='#4fc3f7')
    fig4.update_layout(
        height=400, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title='Anomaly Rate (%)', xaxis_title='',
        xaxis_tickangle=-35,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig4.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig4, use_container_width=True)

    # Summary table
    st.markdown("<div class='section-header'>Detailed Summary Table</div>",
                unsafe_allow_html=True)
    display_df = summary_df.copy()
    display_df['Anomaly Rate (%)'] = display_df['Anomaly Rate (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Avg Anomaly Return'] = display_df['Avg Anomaly Return'].apply(lambda x: f"{x:+.3f}%")
    st.dataframe(display_df, use_container_width=True, height=320)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: Market Event Validation
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Anomaly Detections vs Known Market Events</div>",
                unsafe_allow_html=True)

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
    val_rate  = len(validated) / len(match_df) * 100 if len(match_df) > 0 else 0

    # Validation KPIs
    k1, k2, k3 = st.columns(3)
    for col, label, val, color in [
        (k1, "Anomalies Detected",    str(len(match_df)),          "#4fc3f7"),
        (k2, "Matched to Events",     str(len(validated)),          "#66bb6a"),
        (k3, "Validation Rate",       f"{val_rate:.0f}%,",           "#ce93d8"),
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
        line=dict(color='#4fc3f7', width=1.5),
    ))

    val_dates = pd.to_datetime(validated['Anomaly Date'])
    unval_dates = pd.to_datetime(match_df[~match_df['_validated']]['Anomaly Date'])

    confirmed_df   = df_feat[df_feat.index.isin(val_dates)]
    unconfirmed_df = df_feat[df_feat.index.isin(unval_dates)]

    fig5.add_trace(go.Scatter(
        x=confirmed_df.index, y=confirmed_df['Close'],
        mode='markers', name='Anomaly — Event Confirmed',
        marker=dict(color='#ef5350', size=11, symbol='circle',
                    line=dict(color='#ff8a80', width=1.5)),
    ))
    fig5.add_trace(go.Scatter(
        x=unconfirmed_df.index, y=unconfirmed_df['Close'],
        mode='markers', name='Anomaly — No Event Match',
        marker=dict(color='#ff9800', size=7, symbol='circle', opacity=0.7),
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
                arrowhead=2, arrowcolor='#8b95b0', arrowsize=0.8,
                font=dict(size=10, color='#c5cae9'),
                bgcolor='rgba(30,33,48,0.85)',
                bordercolor='#2d3250', borderwidth=1,
                ay=-40,
            )

    fig5.update_layout(
        height=420, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='right', x=1),
        yaxis_title='Price (KRW)',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig5.update_xaxes(gridcolor='#1e2130')
    fig5.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig5, use_container_width=True)

    # Match table
    st.markdown("<div class='section-header'>Anomaly ↔ Event Match Table</div>",
                unsafe_allow_html=True)
    display_match = match_df.drop(columns=['_validated']).copy()
    st.dataframe(display_match, use_container_width=True, height=300)

    # Category breakdown
    if len(validated) > 0:
        st.markdown("<div class='section-header'>Validated Anomalies by Event Category</div>",
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
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 5: Macro Overview
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Macro Overview</div>", unsafe_allow_html=True)
    df_macro = load_tableau_data().copy()

    fig_macro = make_subplots(rows=4, cols=1, shared_xaxes=True,
                              vertical_spacing=0.04,
                              subplot_titles=["VIX", "USD/KRW", "US 10Y", "Inflation Expectation"])

    fig_macro.add_trace(go.Scatter(x=df_macro['date'], y=df_macro['vix'], name='VIX',
                                   line=dict(color='#4fc3f7')), row=1, col=1)
    fig_macro.add_trace(go.Scatter(x=df_macro['date'], y=df_macro['usd_krw'], name='USD/KRW',
                                   line=dict(color='#ffb74d')), row=2, col=1)
    fig_macro.add_trace(go.Scatter(x=df_macro['date'], y=df_macro['us_10y'], name='US 10Y',
                                   line=dict(color='#81c784')), row=3, col=1)
    fig_macro.add_trace(go.Scatter(x=df_macro['date'], y=df_macro['inflation_exp'],
                                   name='Inflation Exp', line=dict(color='#ce93d8')),
                        row=4, col=1)

    anom_dates = df_macro.loc[df_macro['anomaly_flag'] == 1, 'date']
    for dt in anom_dates:
        fig_macro.add_vline(x=dt, line_width=0.8, line_dash='dot', line_color='#ef5350', opacity=0.6)

    fig_macro.update_layout(
        height=700, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_macro.update_xaxes(gridcolor='#1e2130')
    fig_macro.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig_macro, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 6: Regime Analysis
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>Regime Analysis</div>", unsafe_allow_html=True)
    df_regime = load_tableau_data().copy()

    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(
        x=df_regime['date'], y=df_regime['kospi_price'],
        mode='lines', name='KOSPI Price',
        line=dict(color='#4fc3f7', width=1.5),
    ))

    cusum_dates = df_regime.loc[df_regime['regime_cusum'] == 1, 'date']
    cusum_prices = df_regime.loc[df_regime['regime_cusum'] == 1, 'kospi_price']
    fig_regime.add_trace(go.Scatter(
        x=cusum_dates, y=cusum_prices,
        mode='markers', name='CUSUM Change',
        marker=dict(color='#ff9800', size=7, symbol='x'),
    ))

    # Add regime bands
    regimes = df_regime[['date', 'regime_hmm']].dropna().copy()
    regimes['regime_hmm'] = regimes['regime_hmm'].astype(int)
    if not regimes.empty:
        start_date = regimes.iloc[0]['date']
        current = regimes.iloc[0]['regime_hmm']
        for i in range(1, len(regimes)):
            if regimes.iloc[i]['regime_hmm'] != current:
                end_date = regimes.iloc[i]['date']
                color = 'rgba(76,175,80,0.15)' if current == 1 else 'rgba(239,83,80,0.15)'
                fig_regime.add_vrect(x0=start_date, x1=end_date, fillcolor=color, line_width=0)
                start_date = end_date
                current = regimes.iloc[i]['regime_hmm']
        # last segment
        color = 'rgba(76,175,80,0.15)' if current == 1 else 'rgba(239,83,80,0.15)'
        fig_regime.add_vrect(x0=start_date, x1=regimes.iloc[-1]['date'], fillcolor=color, line_width=0)

    fig_regime.update_layout(
        height=450, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title='KOSPI', margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    )
    fig_regime.update_xaxes(gridcolor='#1e2130')
    fig_regime.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig_regime, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 7: Factor Attribution
# ══════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("<div class='section-header'>Factor Attribution</div>", unsafe_allow_html=True)

    reg_summary = load_regression_summary()
    if not reg_summary.empty:
        fig_reg = go.Figure(go.Bar(
            x=reg_summary['feature'],
            y=reg_summary['coef'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=reg_summary['ci_upper'] - reg_summary['coef'],
                arrayminus=reg_summary['coef'] - reg_summary['ci_lower'],
            ),
            marker_color='#4fc3f7',
        ))
        fig_reg.update_layout(
            height=380, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Coefficient', margin=dict(l=0, r=10, t=10, b=0),
        )
        fig_reg.update_xaxes(gridcolor='#1e2130')
        fig_reg.update_yaxes(gridcolor='#1e2130')
        st.plotly_chart(fig_reg, use_container_width=True)

    shap_values = load_shap_values()
    if not shap_values.empty:
        shap_features = shap_values.drop(columns=['date'], errors='ignore')
        shap_importance = shap_features.abs().mean().sort_values(ascending=False)
        fig_shap = go.Figure(go.Bar(
            x=shap_importance.index,
            y=shap_importance.values,
            marker_color='#ce93d8',
        ))
        fig_shap.update_layout(
            height=320, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Mean |SHAP|', margin=dict(l=0, r=10, t=10, b=0),
        )
        fig_shap.update_xaxes(gridcolor='#1e2130')
        fig_shap.update_yaxes(gridcolor='#1e2130')
        st.plotly_chart(fig_shap, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5080; font-size:12px; padding:8px 0'>
    KOSPI Anomaly Detection · Yoon Hwang · UW–Madison Data Science & Economics · 2025<br>
    Isolation Forest · yfinance · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)