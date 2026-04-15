
### Framework

#### 1. Universe Selection & Liquidity Constraints
In the Indian Cash Market, naked overnight short selling is prohibited. Therefore, the pipeline restricts the Nifty 500 universe exclusively to the ~180 stocks actively traded in the **NSE F&O Segment**. 
* **Operational Edge:** Allows symmetric long/short execution using Current Month Futures.
* **Risk Mitigation:** Isolates the portfolio to the highest liquidity tier, minimizing bid-ask slippage.

#### 2. The Sector-Constrained Cointegration Sieve
Testing all possible pairs creates a massive risk of "Data Mining Bias." The model is constrained to only compare stocks with shared economic drivers (Intra-Sector).
* **Stage 1 (Pearson Correlation):** Preliminary filter for co-movement. Only pairs with r >= 0.80 proceed.
* **Stage 2 (Engle-Granger Test):** Tests the null hypothesis of a unit root in the residuals. Only pairs with a p-value <= 0.05 are accepted, confirming that their price divergence is a stationary I(0) process.

#### 3. Dynamic Spread Engine (Rolling OLS)
The engine estimates the dynamic hedge ratio (Beta) using a 60-day rolling Ordinary Least Squares (OLS) regression:
Spread_t = ln(Y_t) - (Beta * ln(X_t) + Alpha)
* *Logarithmic Transformation:* Using log prices ensures that Beta acts as an elasticity multiplier, immunizing the spread against absolute price level differences.

#### 4. Signal Generation (Z-Score)
The spread is normalized into a rolling Z-Score to identify statistical extremes:
Z_t = (Spread_t - Mean_60) / StdDev_60
* **Entry:** |Z_t| > 2.0 (Initiate Market-Neutral Long/Short Spread).
* **Exit:** |Z_t| <= 0.5 (Mean Reversion achieved; convergence exit).

---

### Execution Protocol

#### Prerequisites
The pipeline requires a standard scientific Python stack:
```bash
pip install pandas numpy statsmodels tqdm