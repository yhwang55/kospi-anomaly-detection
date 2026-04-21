# KOSPI Stock Anomaly Detection (2020–2024)

Detect abnormal trading days across KOSPI blue-chip stocks using Isolation Forest and validate signals against 63 real-world market events.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit)](https://kospi-anomaly-detection-yoonhwang.streamlit.app/) [![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](kospi_anomaly_detection.ipynb) [![Tableau](https://img.shields.io/badge/Dashboard-Tableau-blue?logo=tableau)](https://public.tableau.com/app/profile/yoon.hwang7766/viz/KOSPIAnomalyDetectionMacroFactorAnalysis/KOSPIAnomalyDetectionMacroFactorAnalysis) [![Report](https://img.shields.io/badge/Report-PDF-red?logo=adobeacrobatreader)](KOSPI_Project_Report.pdf)

---

## Overview
KOSPI trading days can exhibit extreme, market-moving behavior that impacts risk, liquidity, and trading performance. This project flags those abnormal days using multivariate features and verifies the signals against curated market events (macro shocks, earnings surprises, geopolitical events).

---

## Key Results

### Model Performance (pseudo-labels: |Z| > 2.0)
| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Isolation Forest | 0.082 | 0.833 | 0.149 |
| Z-score Baseline | 1.000 | 1.000 | 1.000 |

### Market Event Validation
- **Validation rate:** 27 / 61 anomalies matched (44.3%)
- **Return magnitude:** Avg anomaly return **+0.78%** vs normal **−0.02%** → **36.5×** larger moves

### Top 5 Validated Detections
| Date | Return (%) | Event | Category | Impact |
|---|---:|---|---|---|
| 2020-02-24 | -4.05 | KOSPI circuit breaker triggered as COVID fears spike globally | Macro | High |
| 2020-03-11 | -4.58 | Fed emergency rate cut to 0%; KOSPI drops 8% | Macro | High |
| 2020-03-13 | -4.13 | Fed emergency rate cut to 0%; KOSPI drops 8% | Macro | High |
| 2020-03-16 | -2.10 | Fed emergency rate cut to 0%; KOSPI drops 8% | Macro | High |
| 2020-03-17 | -3.27 | KOSPI hits 52-week low (1457); KRW/USD spikes to 1285 | Macro | High |

---

## Features
- Isolation Forest with dynamic thresholding (mean − n×std)
- Cross-stock anomaly comparison across 20 KOSPI leaders
- Market event validation (±2 trading day window)
- Streamlit dashboard with interactive exploration
- SHAP explanations for feature attribution
- Cross-stock clustering signal (3+ stocks anomalous on the same date)

---

## Tech Stack
- **Python**, **pandas**, **numpy**
- **scikit-learn** (Isolation Forest)
- **yfinance** (OHLCV data)
- **Streamlit**, **Plotly**
- **SHAP**, **SciPy** (explanations + statistical testing)
- **TensorFlow/Keras** (LSTM autoencoder baseline)

---

## Project Structure
```
kospi-anomaly-detection/
├── app.py                          # Streamlit dashboard
├── kospi_anomaly_detection.ipynb    # Full analysis notebook
├── kospi_anomaly_detection_appendix.ipynb  # New sections appended as an appendix
├── market_events.csv               # Curated KOSPI market events (63 events)
├── requirements.txt
└── README.md
```

---

## How to Run
```bash
git clone https://github.com/yhwang55/kospi-anomaly-detection
cd kospi-anomaly-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## Business Applications
1. **Risk Monitoring**: Detect extreme market regimes and increase risk controls during clustered anomaly windows.
2. **Trade Surveillance**: Flag unusual volume/price patterns for compliance review.
3. **Portfolio Hedging**: Use cross-stock anomaly clustering as a portfolio-level stress signal.

---

## Limitations & Future Work
- Expand ground-truth labels beyond Z-score pseudo-labels
- Increase event coverage and granularity across sectors
- Extend the LSTM autoencoder with multivariate sequences and sector-specific tuning
- Incorporate real-time market feeds for production monitoring

---

*Yoon Hwang · Data Science & Economics & Information Science · University of Wisconsin–Madison · 2025*
