# NiftyQuantSignals: HMM-Based Factor Timing Engine

An institutional-grade, long-only quantitative asset allocation engine that dynamically rotates factor exposure across the Nifty 500 universe. The core pipeline maps latent macroeconomic states using an unsupervised Hidden Markov Model (HMM) and rebalances capital into optimal quantitative factors (Momentum, Quality, Low-Volatility) while incorporating real-world transaction friction and structural market constraints.

## Framework Architecture

### 1. Data Ingestion & Institutional Constraint Pipeline
* **Universe Ingestion:** Tracks closing prices across the broad **Nifty 500** universe over a **10-year lookback period** (~2,700+ trading days).
* **Missing Data Logging & Structural Alignment:** The data pipeline handles delisted tickers, missing timezone flags, and tracking failures. It dynamically maps downloaded data back to the base matrix using a `.reindex()` buffer, preventing column length mismatches. Missing entries for structural dropouts are left blank (`NaN`).
* **NSE Outlier Sterilization (Circuit Breaker Simulation):** To neutralize massive "fake alpha" distortions caused by unadjusted historical corporate actions (stock splits, demergers, bonus issues) from free API infrastructure, daily asset returns are mathematically capped using a **+/-20% daily circuit-breaker clip** (`.clip(-0.20, 0.20)`). This handles data anomalies, reducing artificial annualized model volatility from 355% down to a realistic risk profile.

### 2. Unsupervised Market Regime Classifier (Rolling Gaussian HMM)
Instead of relying on arbitrary moving averages, the engine isolates underlying market regimes by fitting an unsupervised mathematical model:
* **Feature Matrix:** Prepares a multivariate observation matrix composed of **Daily Log Returns** ($\ln(P_t / P_{t-1})$) and a **5-day rolling Standard Deviation** (Volatility proxy). 
* **State Space Engine:** Fits a **3-State Gaussian Hidden Markov Model (GaussianHMM)** with a full covariance matrix over a strict **756-day rolling window** (~3 years of historical trading data) to guarantee zero **look-ahead bias** in walk-forward environments.
* **Deterministic Sorting & Mapping:** To bypass random initialization noise, states are mathematically ordered by their expected returns ($\mu$). The states are mapped deterministically:
  * $\text{State with Lowest } \mu \rightarrow$ **BEAR Regime**
  * $\text{State with Median } \mu \rightarrow$ **SIDEWAYS Regime**
  * $\text{State with Highest } \mu \rightarrow$ **BULL Regime**

### 3. Dynamic Factor Rotation Matrix
Every **21 trading days** (approx. one business month), the engine re-evaluates the active HMM state and alters the scoring weights across the investment factor matrix:
* **Bull Market (Aggressive Profile):** Allocation favors high beta/trend capture. **70% Momentum** (126-day price return percentile rank) + **30% Low Volatility**.
* **Sideways Market (Neutral Profile):** Allocation favors value defense. **50% Quality** (Inverse 252-day Maximum Drawdown percentile rank) + **50% Low Volatility**.
* **Bear Market (Defensive Profile):** Capital preservation. **100% Low Volatility** (Percentile rank of inverse 60-day price standard deviation, computed directly via `ascending=False` to prevent zero-variance infinity calculation errors).

---

## Portfolio Execution & Simulation Protocol

### Walk-Forward Rolling Backtest
* **Rebalancing Frequency:** 21 Trading Days.
* **Asset Allocation:** Top 10 ranked equities (Top 2% of the universe) are selected at each interval and equally weighted ($w_i = 10\%$).
* **Dependency Injection:** The simulation infrastructure isolates execution logic from modeling logic. Strategy architectures are injected cleanly into the loop as functional parameters to avoid state contamination or circular imports.
* **Real-World Friction Drag:** Incorporates a **10 bps one-way transaction cost** explicitly charged against portfolio turnover matrix adjustments on rebalance days:
$$\text{Cost Drag} = \text{Turnover} \times \frac{\text{Transaction Cost BPs}}{10000}$$

### Performance Metrics & Tear Sheet (10-Year Out-of-Sample Results)
Evaluated against the official, free-float market-cap-weighted **Nifty 500 Index Price Return** (`^CRSLDX`), the engine demonstrates strong alpha generation and risk mitigation:

| Performance Metric | Dynamic HMM Strategy | Official Nifty 500 Benchmark |
| :--- | :---: | :---: |
| **Total Return** | **219.03%** | 133.11% |
| **CAGR (Annualized)** | **16.09%** | 11.50% |
| **Max Drawdown** | **-28.75%** | -38.30% |
| **Annualized Volatility** | **16.30%** | 17.19% |
| **Sharpe Ratio ($R_f = 6\%$)** | **0.62** | 0.32 |
| **Calmar Ratio** | **0.56** | 0.30 |

*Key Takeaway: The engine effectively doubled the risk-adjusted performance (Sharpe 0.62 vs 0.32) of the underlying asset class while reducing peak drawdowns by approximately **955 bps** through timely defensive rotation into low-volatility cohorts before structural market drawdowns realized.*

---

## Dependencies & Installation

Ensure you have a modern scientific Python environment configured. Install required analytical dependencies using the command below:

```bash
pip install pandas numpy yfinance hmmlearn scikit-learn matplotlib