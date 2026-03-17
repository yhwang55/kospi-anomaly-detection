# KOSPI Anomaly Detection Dashboard

An interactive financial anomaly detection system built with **Isolation Forest**, deployed as a Streamlit web app.

**[🚀 Live Demo →](https://your-app.streamlit.app)** &nbsp;|&nbsp; **[📓 Jupyter Notebook →](kospi_anomaly_detection_v2.ipynb)**

---

## Overview

Detects statistically anomalous trading days across **20 KOSPI blue-chip stocks** (2020–2024) using unsupervised machine learning, and validates detections against real-world market events.

## Features

| Tab | Content |
|-----|---------|
| 📊 Anomaly Detection | Interactive price chart with flagged anomaly dates, volume overlay, IF score timeline |
| 🔬 Feature Analysis | Distribution comparison (normal vs anomaly) + feature separation power (Cohen's d) |
| 🏢 Cross-Stock Comparison | Anomaly rates across all 20 stocks with dynamic threshold |
| 📅 Market Event Validation | Anomaly ↔ event matching (±2 trading days), validation rate, category breakdown |

All parameters (number of trees, threshold σ, stock selection) are adjustable via sidebar in real time.

## Tech Stack

- **ML Model:** Isolation Forest (`scikit-learn`) with dynamic `mean − n×std` threshold
- **Data:** `yfinance` — 5-year OHLCV, 20 KOSPI stocks (~24,580 data points)
- **Features:** Daily return, MA5/MA20 deviation, volume Z-score, 5-day volatility, PV signal
- **Visualization:** `Plotly` interactive charts
- **App:** `Streamlit`

## Run Locally

```bash
git clone https://github.com/your-username/kospi-anomaly-detection
cd kospi-anomaly-detection
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
kospi-anomaly-detection/
├── app.py                          # Streamlit dashboard
├── kospi_anomaly_detection_v2.ipynb  # Full analysis notebook
├── market_events.csv               # Curated KOSPI market events (63 events)
├── requirements.txt
└── README.md
```

## Key Results

- Isolation Forest flagged **~5% of trading days** as anomalous across the KOSPI universe
- Anomaly days show average absolute returns **3–5× higher** than normal days
- Top detections align with: 2024-08-05 carry trade unwind (−10.3%), 2024-03-20 AI chip surge (+5.6%), 2023-04-07 Samsung production cut (+4.3%)
- Model validated against 63 curated KOSPI market events (Bloomberg, Yonhap News, IR releases)

---

*Yoon Hwang · Data Science & Economics · University of Wisconsin–Madison · 2025*
