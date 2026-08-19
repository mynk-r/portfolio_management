# Indian Quant Portfolio Intelligence

### Classical Portfolio Optimization vs Machine Learning for Indian Equities

An end-to-end quantitative finance project that investigates whether **machine-learning-based return forecasts can improve portfolio construction for Indian equities** compared with classical optimization methods.

The project combines market-data engineering, risk analysis, portfolio optimization, walk-forward backtesting, ML forecasting, transaction costs, and NIFTY 50 benchmarking.

---

## Research Question

> **Can ML-based forward-return forecasts improve the risk-adjusted performance of Indian equity portfolios relative to classical portfolio optimization?**

The project compares:

**Classical**
- Equal Weight
- Minimum Variance
- Maximum Sharpe
- Risk Parity

**Machine Learning**
- Random Forest
- XGBoost

**Benchmark**
- NIFTY 50

---

## Project Flow

```text
Yahoo Finance / yfinance
          ↓
     Data Collection
          ↓
    Data Cleaning
          ↓
    Risk Analysis
          ↓
 ┌────────┴─────────┐
 │                  │
 ▼                  ▼
Classical Models   ML Models
 │                  │
 │            Random Forest
 │            XGBoost
 │                  │
 └────────┬─────────┘
          ↓
 Portfolio Construction
          ↓
 Walk-Forward Backtest
          ↓
 Transaction Costs
          ↓
 NIFTY 50 Benchmark
          ↓
 Performance & Risk Analysis
          ↓
 ML Explainability
          ↓
 Portfolio Allocation Analysis
          ↓
 Future Streamlit Dashboard
```

---

## Key Methodology

### Portfolio Optimization

- Equal Weight
- Minimum Variance
- Maximum Sharpe
- Risk Parity
- Long-only constraints
- Maximum 15% allocation per stock

### ML Prediction

Models predict **21-trading-day forward returns** using market-derived features:

- 5/21/63/126-day momentum
- 21/63-day volatility
- Price vs moving averages
- RSI
- Drawdown
- Skewness
- Kurtosis

No macroeconomic variables are used.

### Backtesting

A walk-forward framework is used with approximately:

- **504 trading-day training window**
- **21-trading-day forward test period**
- Monthly rebalancing
- Transaction-cost modelling

This is designed to reduce look-ahead bias and provide a more realistic out-of-sample evaluation.

---

## Risk & Performance Evaluation

Strategies are evaluated using:

- CAGR
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- VaR
- CVaR
- Alpha / Beta vs NIFTY 50
- Tracking Error
- Information Ratio

Additional robustness analysis includes:

- 12-month rolling Sharpe
- Rolling volatility
- Rolling alpha
- Portfolio concentration
- Effective number of stocks
- Portfolio weight evolution

---

## ML Explainability

The project evaluates feature importance for both:

- Random Forest
- XGBoost

This helps identify which market-derived characteristics are most influential in the models' return forecasts.

Feature importance is interpreted as **model importance, not causal evidence**.

---

## Repository Structure

```text
.
├── 00_data_gen.ipynb
├── 01_risk_analysis.ipynb
├── 02_portfolio.ipynb
├── 03_walk_forward_backtest.ipynb
├── 04_ML.ipynb
├── 05_final_eval.ipynb
├── 06_adv_analysis.ipynb
│
├── data/
│   ├── raw/
│   │   ├── adjusted_close_prices.csv
│   │   ├── daily_returns.csv
│   │   ├── data_summary.csv
│   │   ├── indian_equities_raw.csv
│   │   ├── indian_market_raw.csv
│   │   ├── market_prices.csv
│   │   └── master_market_dataset.csv
│   │
│   └── processed/
│       ├── backtest/
│       ├── final_backtest/
│       ├── final_evaluation/
│       ├── optimization/
│       ├── risk_summary.csv
│       ├── correlation_matrix.csv
│       ├── covariance matrices
│       ├── return datasets
│       └── rolling risk datasets
│
└── outputs/
    ├── advanced_analysis/
    └── figures/
```

---

## Notebook Pipeline

| Notebook | Purpose |
|---|---|
| `00_data_gen.ipynb` | Data collection and preprocessing |
| `01_risk_analysis.ipynb` | Returns, risk, correlation and covariance |
| `02_portfolio.ipynb` | Classical portfolio optimization |
| `03_walk_forward_backtest.ipynb` | Out-of-sample classical backtesting |
| `04_ML.ipynb` | ML return prediction and portfolio construction |
| `05_final_eval.ipynb` | Strategy comparison and NIFTY benchmarking |
| `06_adv_analysis.ipynb` | Explainability, rolling analysis and allocation |

---

## Data

Market data is collected programmatically using **Yahoo Finance through `yfinance`**.

The pipeline generates:

- Adjusted close prices
- Daily returns
- Market/index data
- Portfolio and risk datasets

For production investment use, data should be cross-validated with a professional market-data provider.

---

## Visualizations

Current analysis includes:

- Cumulative strategy performance
- Drawdown comparison
- Risk-return profile
- Sharpe comparison
- Rolling Sharpe
- Rolling volatility
- Rolling alpha vs NIFTY
- ML feature importance
- Portfolio allocation
- Portfolio weight evolution

---

## Key Design Principle

The project does **not** assume that ML is superior.

The objective is to determine empirically whether ML provides incremental value after:

```text
Prediction
→ Portfolio Optimization
→ Out-of-Sample Testing
→ Transaction Costs
→ Risk Evaluation
→ Benchmark Comparison
```

A classical strategy outperforming ML is therefore a valid research result.

---

## Tech Stack

**Python · Pandas · NumPy · SciPy · scikit-learn · XGBoost · yfinance · Matplotlib · Jupyter**

**Planned:** Streamlit dashboard

---

## Current Status

- [x] Automated data pipeline
- [x] Risk analysis
- [x] Classical optimization
- [x] Walk-forward backtesting
- [x] Transaction costs
- [x] Random Forest
- [x] XGBoost
- [x] NIFTY 50 benchmarking
- [x] Risk & performance evaluation
- [x] ML explainability
- [x] Rolling analysis
- [x] Portfolio allocation analysis
- [x] Visualization layer
- [ ] Streamlit dashboard
- [ ] Final GitHub presentation

---

## Future Dashboard

The Streamlit application will provide:

1. **Executive Summary** — strategy performance and NIFTY comparison
2. **Portfolio Analytics** — allocations and concentration
3. **Performance** — returns, drawdowns and rolling risk
4. **ML Intelligence** — model comparison and feature importance

---

## Disclaimer

This is a quantitative research and educational project, **not investment advice**. Historical backtest performance does not guarantee future results.
