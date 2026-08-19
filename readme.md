# Indian Quant Portfolio Intelligence

## Classical Portfolio Optimization vs Machine Learning for Indian Equities

> **Research question:** Can machine-learning-based return forecasts improve risk-adjusted portfolio construction for Indian equities relative to classical portfolio optimization, after accounting for transaction costs and downside risk?

---

## 1. Project Overview

**Indian Quant Portfolio Intelligence** is an end-to-end quantitative investment research project focused on portfolio construction using Indian equities.

The project combines:

- Automated market-data collection using **Yahoo Finance / `yfinance`**
- Indian equity return and risk analysis
- Correlation and covariance modelling
- Rolling risk analytics
- Classical portfolio optimization
- Walk-forward out-of-sample backtesting
- Transaction-cost modelling
- Machine-learning-based forward return prediction
- Random Forest and XGBoost portfolio strategies
- NIFTY 50 benchmark evaluation
- Downside-risk analysis using VaR and CVaR
- Rolling performance and alpha analysis
- ML feature-importance analysis
- Portfolio concentration and allocation analysis
- Publication-ready performance visualizations

The project deliberately avoids macroeconomic variables and focuses on **market-derived information available directly from equity prices and market data**.

The eventual dashboard layer will be added separately using Streamlit.

---

# 2. Why This Project?

A conventional stock-prediction project asks:

> "Can I predict whether a stock will go up?"

This project asks a more useful investment question:

> "Can predicted information improve the construction of a diversified portfolio, and does that improvement survive realistic out-of-sample testing and transaction costs?"

That distinction is central to the project.

A model can have reasonable prediction accuracy but still fail as an investment strategy because:

- prediction errors can be economically expensive,
- portfolio optimization can amplify model errors,
- concentrated portfolios can create large drawdowns,
- frequent trading can destroy returns,
- and a strategy must be compared with strong classical benchmarks.

Therefore, ML is treated as a **decision-support layer**, not as the entire investment thesis.

---

# 3. Core Research Question

The flagship research question is:

> **Does machine-learning-based forward return forecasting improve risk-adjusted portfolio construction for Indian equities relative to classical portfolio optimization?**

The project therefore compares:

### Classical strategies

1. Equal Weight
2. Minimum Variance
3. Maximum Sharpe
4. Risk Parity

### Machine-learning strategies

5. Random Forest
6. XGBoost

### Benchmark

7. NIFTY 50

All portfolio strategies are evaluated on a common out-of-sample period wherever possible.

---

# 4. Project Architecture

```text
                         YAHOO FINANCE / YFINANCE
                                  │
                                  ▼
                         DATA COLLECTION LAYER
                                  │
                                  ▼
                      Indian Equity Market Dataset
                                  │
                                  ▼
                         DATA PREPROCESSING
                                  │
                                  ▼
                  ┌───────────────┴───────────────┐
                  │                               │
                  ▼                               ▼
             PRICE DATA                      MARKET DATA
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                         RETURN ENGINEERING
                                  │
                                  ▼
                         RISK ANALYSIS
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
                  ▼                               ▼
        Classical Portfolio              ML Feature Engine
           Optimization                         │
                  │                     ┌────────┴────────┐
                  │                     ▼                 ▼
                  │                Random Forest       XGBoost
                  │                     │                 │
                  └──────────────┬──────┴─────────────────┘
                                 ▼
                      PORTFOLIO CONSTRUCTION
                                 │
                                 ▼
                     WALK-FORWARD BACKTEST
                                 │
                                 ▼
                       TRANSACTION COSTS
                                 │
                                 ▼
                         NIFTY 50 BENCHMARK
                                 │
                                 ▼
                    PERFORMANCE & RISK ANALYSIS
                                 │
                  ┌──────────────┴───────────────┐
                  ▼                              ▼
          Portfolio Analytics              ML Explainability
                  │                              │
                  ▼                              ▼
        Rolling Performance              Feature Importance
        Drawdowns / CVaR                 Allocation Analysis
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                          VISUALIZATION LAYER
                                 │
                                 ▼
                         FUTURE STREAMLIT APP
```

---

# 5. End-to-End Project Flow

## Stage 1 — Data Generation

**Notebook:** `00_data_gen.ipynb`

Market data is collected programmatically rather than manually downloaded.

The project uses Yahoo Finance through the Python `yfinance` ecosystem for:

- Indian equity price data
- Adjusted close prices
- Market index data
- Additional market series used in the project

The raw data is consolidated into reusable CSV datasets.

### Key raw files

```text
data/raw/
├── adjusted_close_prices.csv
├── daily_returns.csv
├── data_summary.csv
├── indian_equities_raw.csv
├── indian_market_raw.csv
├── market_prices.csv
└── master_market_dataset.csv
```

The purpose of this layer is to make the research pipeline reproducible.

---

# 6. Stage 2 — Risk Analysis

**Notebook:** `01_risk_analysis.ipynb`

The second stage establishes the statistical characteristics of the investment universe before any optimization or ML is performed.

Key analyses include:

### Return analysis

- Daily returns
- Weekly returns
- Monthly returns
- Cumulative returns

### Risk analysis

- Rolling volatility
- Rolling Sharpe ratio
- Rolling beta
- Drawdown

### Dependence analysis

- Correlation matrix
- Covariance matrix
- Annualized covariance matrix
- EWMA covariance

### Risk outputs

```text
data/processed/
├── annualized_covariance_matrix.csv
├── correlation_matrix.csv
├── cumulative_returns.csv
├── daily_covariance_matrix.csv
├── drawdown.csv
├── ewma_covariance_matrix.csv
├── monthly_returns.csv
├── risk_summary.csv
├── rolling_beta_60d.csv
├── rolling_sharpe_60d.csv
├── rolling_volatility_60d.csv
└── weekly_returns.csv
```

This stage answers:

> How risky are the individual securities, and how do they interact with each other?

---

# 7. Stage 3 — Classical Portfolio Optimization

**Notebook:** `02_portfolio.ipynb`

This stage establishes the classical quantitative-finance benchmarks.

The project constructs four portfolios.

## 7.1 Equal Weight

Each security receives the same allocation:

\[
w_i = \frac{1}{N}
\]

This serves as a simple and robust baseline.

---

## 7.2 Minimum Variance

The optimizer minimizes:

\[
\sigma_p^2 = w^T \Sigma w
\]

subject to:

\[
\sum_i w_i = 1
\]

and portfolio constraints.

The project uses a long-only framework with a maximum individual-stock weight.

---

## 7.3 Maximum Sharpe

The optimizer maximizes:

\[
Sharpe_p =
\frac{w^T\mu-r_f}
{\sqrt{w^T\Sigma w}}
\]

where:

- \(w\) = portfolio weights
- \(\mu\) = expected returns
- \(\Sigma\) = covariance matrix
- \(r_f\) = risk-free rate assumption

---

## 7.4 Risk Parity

Risk Parity attempts to distribute portfolio risk more evenly across assets.

For each asset:

\[
RC_i =
\frac{w_i(\Sigma w)_i}
{\sigma_p}
\]

The optimizer attempts to make risk contributions approximately equal.

---

# 8. Portfolio Constraints

To avoid unrealistic portfolios, the optimization framework uses:

```text
Long-only
No short selling
Weights sum to 100%
Maximum individual stock weight = 15%
```

These constraints reduce the possibility of an optimizer creating an unrealistic highly concentrated portfolio.

---

# 9. Stage 4 — Walk-Forward Backtesting

**Notebook:** `03_walk_forward_backtest.ipynb`

This is one of the most important components of the project.

The strategies are not evaluated using a simple in-sample backtest.

Instead, the project uses **walk-forward testing**.

### Concept

```text
Historical Window
       │
       ▼
Optimize portfolio
       │
       ▼
Future unseen period
       │
       ▼
Realized return
       │
       ▼
Move window forward
       │
       ▼
Repeat
```

The current configuration uses approximately:

- **504 trading days** of historical training information
- **21 trading days** of forward evaluation
- approximately monthly rebalancing
- transaction costs at each rebalance

This helps reduce look-ahead bias and gives a more realistic estimate of out-of-sample performance.

---

# 10. Transaction Costs

Portfolio performance is not evaluated purely on gross returns.

The backtest applies a transaction-cost assumption based on portfolio turnover:

\[
Cost_t =
Turnover_t \times TransactionCost
\]

The default modelling assumption is:

```text
Transaction cost = 0.10%
```

This parameter can be changed in the configuration.

The purpose is not to perfectly reproduce every brokerage fee, tax and market-impact component, but to prevent the strategy from receiving unrealistically free trading.

---

# 11. Stage 5 — Machine Learning

**Notebook:** `04_ML.ipynb`

The ML layer predicts **forward 21-trading-day stock returns**.

The target is:

\[
Target_t =
\frac{P_{t+21}}{P_t}-1
\]

The model therefore attempts to estimate the next approximately one-month return from information available at time \(t\).

---

# 12. ML Feature Engineering

The ML models use market-derived features.

## Momentum

- 5-day return
- 21-day return
- 63-day return
- 126-day return

## Volatility

- 21-day volatility
- 63-day volatility

## Trend

- Price vs 21-day moving average
- Price vs 63-day moving average
- Price vs 126-day moving average

## Technical state

- RSI(14)

## Downside / regime information

- 126-day drawdown

## Distribution characteristics

- 63-day skewness
- 63-day kurtosis

Importantly, the project does **not** use macroeconomic variables in the ML model.

---

# 13. Machine Learning Models

Two models are used.

## Random Forest

Random Forest provides a nonlinear ensemble baseline.

It is useful for capturing:

- nonlinear relationships,
- interactions between technical features,
- regime-dependent relationships.

---

## XGBoost

XGBoost is the primary boosted-tree model.

It is particularly suitable for this problem because the feature set is:

- tabular,
- relatively low dimensional,
- nonlinear,
- potentially interaction-heavy.

The project does not use a large collection of models merely to create a model zoo.

The objective is to determine whether a small number of strong models can provide economically useful portfolio signals.

---

# 14. From Predictions to Portfolio Weights

The ML models do not directly produce a "buy/sell" signal.

Instead:

```text
Market features
       ↓
ML model
       ↓
Predicted forward returns
       ↓
Constrained portfolio optimizer
       ↓
Portfolio weights
```

The predicted returns become the expected-return input to a portfolio optimizer.

The ML portfolio therefore attempts to maximize the predicted risk-adjusted return while respecting the same portfolio constraints used by the classical strategies.

This is an important distinction:

> The ML model forecasts; the portfolio optimizer allocates.

---

# 15. Avoiding Look-Ahead Bias

The project uses walk-forward modelling.

At every rebalance:

1. Only historical observations before the rebalance date are used.
2. The ML model is trained using that historical window.
3. Predictions are generated for the current decision date.
4. Portfolio weights are constructed.
5. The following 21 trading days are treated as unseen data.
6. Realized returns are recorded.
7. The window moves forward.

This prevents future returns from being used to construct current portfolio decisions.

---

# 16. Stage 6 — Final Evaluation

**Notebook:** `05_final_eval.ipynb`

The final evaluation compares all strategies against the NIFTY 50 benchmark.

### Strategies

```text
NIFTY 50
Equal Weight
Minimum Variance
Maximum Sharpe
Risk Parity
Random Forest
XGBoost
```

The comparison includes:

### Return metrics

- Total Return
- CAGR

### Risk metrics

- Annualized Volatility
- Maximum Drawdown
- VaR
- CVaR

### Risk-adjusted metrics

- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Benchmark-relative metrics

- Beta to NIFTY 50
- Annualized Alpha
- Excess Return
- Tracking Error
- Information Ratio

---

# 17. Why CVaR?

Traditional volatility treats upside and downside variation symmetrically.

For a risk-management-oriented portfolio project, downside tail risk is also important.

The project therefore includes:

### VaR

The 5th percentile of daily returns:

\[
VaR_{95} = Q_{0.05}(R)
\]

### CVaR

The average return conditional on being in the worst 5% of observations:

\[
CVaR_{95}
=
E[R\mid R\leq VaR_{95}]
\]

CVaR provides a more informative view of extreme downside outcomes than volatility alone.

---

# 18. Benchmarking Against NIFTY 50

The NIFTY 50 is used as the primary Indian equity benchmark.

The evaluation asks:

> Does the portfolio provide better risk-adjusted performance than simply holding the broad Indian market?

This allows the project to move beyond:

> "My model made X%."

and toward:

> "My strategy generated X% CAGR with Y Sharpe and Z% maximum drawdown relative to the NIFTY benchmark."

---

# 19. Stage 7 — Advanced Analysis

**Notebook:** `06_adv_analysis.ipynb`

The final analytical notebook focuses on three areas.

---

## 19.1 ML Explainability

Feature importance is extracted from:

- Random Forest
- XGBoost

The project compares the relative importance of:

- momentum,
- volatility,
- trend,
- RSI,
- drawdown,
- distribution characteristics.

Feature importance is interpreted as **model importance**, not causal economic evidence.

---

## 19.2 Rolling Performance

Static Sharpe ratios can hide changes in strategy behaviour.

Therefore the project examines:

- 12-month rolling Sharpe
- 12-month rolling volatility
- 12-month rolling alpha vs NIFTY

This helps determine whether performance is persistent or concentrated in a particular market regime.

---

## 19.3 Portfolio Allocation Analysis

Portfolio weights are analysed using:

- average stock allocation,
- top stock allocations,
- maximum position size,
- number of meaningful positions,
- Herfindahl-Hirschman Index (HHI),
- effective number of stocks,
- portfolio weight evolution.

The effective number of stocks is:

\[
N_{effective}
=
\frac{1}{\sum_i w_i^2}
\]

This provides a useful measure of effective diversification.

---

# 20. Visualization Layer

The project generates visual outputs for the major analytical questions.

### Performance

- Cumulative portfolio wealth
- Drawdown comparison
- Risk-return profile
- Sharpe ratio comparison

### Robustness

- Rolling Sharpe
- Rolling volatility
- Rolling alpha vs NIFTY

### ML explainability

- Random Forest feature importance
- XGBoost feature importance
- RF vs XGBoost feature comparison

### Portfolio construction

- Top stock allocations
- Portfolio weight evolution

These figures are stored under:

```text
outputs/
├── advanced_analysis/
└── figures/
```

---

# 21. Repository Structure

The current repository is organized as follows:

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
│   │
│   ├── processed/
│   │   │
│   │   ├── annualized_covariance_matrix.csv
│   │   ├── correlation_matrix.csv
│   │   ├── cumulative_returns.csv
│   │   ├── daily_covariance_matrix.csv
│   │   ├── drawdown.csv
│   │   ├── ewma_covariance_matrix.csv
│   │   │
│   │   ├── backtest/
│   │   │
│   │   ├── final_backtest/
│   │   │
│   │   ├── final_evaluation/
│   │   │
│   │   ├── monthly_returns.csv
│   │   ├── optimization/
│   │   ├── risk_summary.csv
│   │   ├── rolling_beta_60d.csv
│   │   ├── rolling_sharpe_60d.csv
│   │   ├── rolling_volatility_60d.csv
│   │   └── weekly_returns.csv
│   │
│   └── raw/
│       │
│       ├── adjusted_close_prices.csv
│       ├── daily_returns.csv
│       ├── data_summary.csv
│       ├── indian_equities_raw.csv
│       ├── indian_market_raw.csv
│       ├── market_prices.csv
│       └── master_market_dataset.csv
│
└── outputs/
    │
    ├── advanced_analysis/
    └── figures/
```

The repository will later be extended with the interactive dashboard and final documentation.

---

# 22. Notebook Responsibilities

| Notebook | Purpose |
|---|---|
| `00_data_gen.ipynb` | Data collection, cleaning and master dataset creation |
| `01_risk_analysis.ipynb` | Return, volatility, correlation, covariance and drawdown analysis |
| `02_portfolio.ipynb` | Classical portfolio construction and optimization |
| `03_walk_forward_backtest.ipynb` | Out-of-sample classical strategy backtesting |
| `04_ML.ipynb` | ML return forecasting and ML portfolio construction |
| `05_final_eval.ipynb` | Final comparison against NIFTY 50 |
| `06_adv_analysis.ipynb` | Explainability, rolling performance and allocation analysis |

This separation keeps the research pipeline easy to audit and reproduce.

---

# 23. Data Sources

## Primary source — Yahoo Finance

Market data is collected programmatically through the `yfinance` Python ecosystem.

The project primarily uses:

- Adjusted close prices
- Indian equity price series
- NIFTY 50 market data
- Other market-price series required by the data-generation pipeline

The exact securities and date ranges are defined in the data-generation notebook and generated datasets.

### Why Yahoo Finance?

- Programmatic access
- Reproducible data collection
- Easy integration with Python
- Suitable for research and educational portfolio analysis
- Avoids manually maintaining downloaded datasets

For production investment use, market data should be replaced or cross-validated with a licensed institutional data provider.

---

# 24. Data Pipeline

```text
Yahoo Finance
      │
      ▼
yfinance
      │
      ▼
Raw CSV files
      │
      ▼
Cleaning / alignment
      │
      ▼
Adjusted close prices
      │
      ▼
Daily returns
      │
      ▼
Risk & feature datasets
      │
      ▼
Portfolio / ML engines
```

---

# 25. Technology Stack

### Programming

- Python

### Data

- pandas
- NumPy

### Financial modelling

- SciPy
- yfinance

### Machine Learning

- scikit-learn
- XGBoost

### Visualization

- Matplotlib

### Research environment

- Jupyter Notebook

### Future deployment

- Streamlit

---

# 26. Key Quantitative Methods

The project incorporates:

- Historical return analysis
- Volatility estimation
- Correlation analysis
- Covariance estimation
- EWMA covariance
- Portfolio optimization
- Mean-variance optimization
- Maximum Sharpe optimization
- Risk Parity
- Walk-forward validation
- Transaction-cost modelling
- Forward return prediction
- Ensemble tree models
- Feature importance
- Drawdown analysis
- VaR
- CVaR
- Alpha/Beta analysis
- Tracking Error
- Information Ratio
- Portfolio concentration analysis

---

# 27. Research Design

The project follows a layered research design.

### Layer 1 — Understand the market

Before modelling, examine:

- returns,
- volatility,
- correlation,
- covariance,
- drawdowns.

### Layer 2 — Establish classical benchmarks

Build portfolios using:

- Equal Weight,
- Minimum Variance,
- Maximum Sharpe,
- Risk Parity.

### Layer 3 — Introduce machine learning

Use market-derived features to predict forward returns using:

- Random Forest
- XGBoost

### Layer 4 — Convert predictions into decisions

Use ML predictions as expected-return signals for constrained portfolio optimization.

### Layer 5 — Test out-of-sample

Use walk-forward evaluation rather than a single train/test split.

### Layer 6 — Account for implementation friction

Apply transaction costs and evaluate turnover.

### Layer 7 — Benchmark

Compare against NIFTY 50.

### Layer 8 — Investigate robustness

Study:

- rolling Sharpe,
- rolling volatility,
- rolling alpha,
- drawdowns,
- CVaR,
- concentration,
- feature importance.

---

# 28. What Makes This a Flagship Project?

The project is designed around **investment decision-making rather than model accuracy alone**.

### 1. Indian-market focus

The investment universe is explicitly based on Indian equities rather than generic US-market datasets.

### 2. Strong classical benchmarks

ML is not compared against a naive prediction baseline. It must compete with:

- Minimum Variance
- Maximum Sharpe
- Risk Parity
- Equal Weight

### 3. Walk-forward validation

The project avoids relying solely on in-sample performance.

### 4. Transaction costs

Trading friction is incorporated into portfolio evaluation.

### 5. Risk-aware evaluation

The project evaluates:

- Sharpe,
- Sortino,
- Maximum Drawdown,
- VaR,
- CVaR,
- Calmar.

### 6. Benchmark-relative analysis

Strategies are evaluated against NIFTY 50.

### 7. Explainability

The ML layer includes feature-importance analysis.

### 8. Portfolio-level ML

The project does not stop at:

> "XGBoost predicts returns."

It continues through:

> Prediction → Optimization → Allocation → Backtest → Risk Evaluation.

That is the central design feature of the project.

---

# 29. Important Limitations

This project is a research and educational portfolio-management system, not a production trading system.

### Data limitations

Yahoo Finance data can contain:

- missing observations,
- corporate-action adjustments,
- historical changes,
- survivorship-related issues,
- ticker inconsistencies.

### Survivorship bias

If the equity universe is defined using currently available securities, historical analysis may suffer from survivorship bias.

A production-grade system should maintain point-in-time constituent histories.

### Transaction costs

The model uses a simplified transaction-cost assumption and does not fully model:

- bid-ask spread,
- market impact,
- brokerage-specific fees,
- securities transaction tax,
- exchange charges,
- slippage,
- taxes.

### Risk-free rate

The initial modelling framework uses a simplified risk-free assumption. A production research system should use an appropriate Indian risk-free series consistently.

### Model risk

ML models can overfit historical relationships that do not persist.

### Backtest risk

Strong historical performance does not guarantee future performance.

---

# 30. How to Interpret the Results

The objective is **not** to automatically declare the ML model the winner.

A valid conclusion could be any of the following:

### Case A — ML wins

If XGBoost provides higher risk-adjusted returns while maintaining reasonable drawdown and turnover:

> ML provides evidence of incremental value over classical portfolio optimization.

### Case B — Classical optimization wins

If Maximum Sharpe or Risk Parity outperforms ML after transaction costs:

> The ML models do not provide sufficient incremental economic value over classical portfolio construction.

### Case C — ML improves risk, not return

If ML produces similar CAGR but materially lower drawdown or CVaR:

> ML may provide value primarily through improved risk management rather than higher raw returns.

All three are legitimate research outcomes.

---

# 31. Reproducibility

The project is designed so that the pipeline can be rerun from the beginning.

Recommended order:

```text
00_data_gen.ipynb
        ↓
01_risk_analysis.ipynb
        ↓
02_portfolio.ipynb
        ↓
03_walk_forward_backtest.ipynb
        ↓
04_ML.ipynb
        ↓
05_final_eval.ipynb
        ↓
06_adv_analysis.ipynb
```

Running notebooks in this order regenerates the major datasets and analytical outputs.

---

# 32. Future Dashboard

The interactive dashboard is intentionally separated from the research notebooks.

Planned Streamlit interface:

```text
HOME
│
├── Executive Summary
│   ├── Best Strategy
│   ├── CAGR
│   ├── Sharpe
│   ├── Maximum Drawdown
│   └── NIFTY Comparison
│
├── Portfolio Analytics
│   ├── Portfolio Weights
│   ├── Risk Contribution
│   ├── Concentration
│   └── Allocation Evolution
│
├── Performance
│   ├── Cumulative Returns
│   ├── Drawdowns
│   ├── Rolling Sharpe
│   ├── Rolling Volatility
│   └── Risk-Return
│
└── ML Intelligence
    ├── RF vs XGBoost
    ├── Feature Importance
    ├── Predicted Returns
    └── ML vs Classical
```

This will turn the research pipeline into an interactive portfolio-intelligence application.

---

# 33. Suggested Dashboard Narrative

The eventual application should not simply display charts.

It should answer four questions:

### Question 1

**How did the strategies perform?**

→ Cumulative wealth, CAGR and benchmark comparison.

### Question 2

**How much risk did they take?**

→ Volatility, drawdown, CVaR and rolling Sharpe.

### Question 3

**What is the portfolio actually holding?**

→ Allocation, concentration and weight evolution.

### Question 4

**Does ML add value?**

→ RF vs XGBoost vs classical optimization.

This keeps the dashboard focused on investment decisions rather than visual decoration.

---

# 34. Future Improvements

Potential extensions for a production-grade version include:

- Point-in-time NIFTY constituent data
- Explicit risk-free-rate series
- Sector constraints
- Sector exposure monitoring
- More realistic transaction-cost modelling
- Slippage and market-impact modelling
- Regime-dependent models
- Probability forecasts
- Purged / embargoed validation for overlapping targets
- Hyperparameter tuning using strictly time-aware validation
- Deflated Sharpe Ratio
- Probability of Backtest Overfitting
- Portfolio turnover constraints
- Live data refresh
- Scheduled portfolio monitoring

These are deliberately outside the current flagship scope to avoid unnecessary model complexity.

---

# 35. Project Status

### Completed

- [x] Automated market-data pipeline
- [x] Indian equity dataset
- [x] Return engineering
- [x] Risk analysis
- [x] Correlation and covariance analysis
- [x] EWMA covariance
- [x] Classical portfolio optimization
- [x] Walk-forward backtesting
- [x] Transaction-cost modelling
- [x] Random Forest portfolio
- [x] XGBoost portfolio
- [x] NIFTY 50 benchmarking
- [x] VaR / CVaR analysis
- [x] Alpha / Beta analysis
- [x] Rolling performance analysis
- [x] ML feature importance
- [x] Portfolio concentration analysis
- [x] Portfolio allocation analysis
- [x] Performance visualizations

### Planned

- [ ] Streamlit dashboard
- [ ] Final GitHub presentation
- [ ] Architecture diagram
- [ ] Dashboard screenshots
- [ ] Final research conclusions

---

# 36. Resume-Ready Description

### Short version

**Indian Quant Portfolio Intelligence — Classical Optimization vs ML**

Built an end-to-end quantitative portfolio research system for Indian equities using `yfinance`, Python and machine learning; compared Equal Weight, Minimum Variance, Maximum Sharpe, Risk Parity, Random Forest and XGBoost portfolios through walk-forward backtesting with transaction costs, NIFTY 50 benchmarking, drawdown, VaR/CVaR, alpha-beta and rolling risk analysis.

### Stronger one-line version

> Developed an out-of-sample quantitative portfolio optimization framework for Indian equities integrating classical risk-based allocation with Random Forest/XGBoost return forecasts, transaction-cost-aware walk-forward backtesting and NIFTY 50 benchmark evaluation.

---

# 37. Final Takeaway

This project is fundamentally about one question:

> **Can information extracted from historical Indian equity markets improve portfolio decisions beyond established quantitative allocation methods?**

The answer is determined empirically rather than assumed.

The complete pipeline is:

```text
DATA
  ↓
RISK
  ↓
CLASSICAL OPTIMIZATION
  ↓
WALK-FORWARD TESTING
  ↓
MACHINE LEARNING
  ↓
PORTFOLIO CONSTRUCTION
  ↓
TRANSACTION COSTS
  ↓
NIFTY BENCHMARK
  ↓
RISK & ROBUSTNESS
  ↓
EXPLAINABILITY
  ↓
DASHBOARD
```

The project therefore combines **quantitative finance, portfolio risk management, machine learning, statistical evaluation and financial analytics** into a single end-to-end research system.
