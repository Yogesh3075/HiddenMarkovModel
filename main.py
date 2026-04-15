"""
NiftyQuantSignals: HMM-Based Factor Timing Engine
================================================
Logic: Dynamically rotates QVMQ factor weights based on
       the Hidden Markov Model state.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# 1. HMM REGIME CLASSIFIER (Rolling to avoid Look-ahead Bias)
def get_hmm_state(price_series):
    # Prepare Features: Returns and Volatility
    ret = np.log(price_series / price_series.shift(1)).fillna(0)
    vol = ret.rolling(5).std().fillna(0)
    obs = np.column_stack([ret, vol])

    # Fit HMM
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    model.fit(obs)
    states = model.predict(obs)

    # Map States to Logic (Identify which state is which)
    # State with highest mean return = Bull (1)
    # State with highest volatility = Bear (2)
    # The remaining state = Sideways (0)
    means = model.means_[:, 0]
    vols = model.means_[:, 1]

    bull_state = np.argmax(means)
    bear_state = np.argmax(vols)
    sideways_state = [i for i in range(3) if i not in [bull_state, bear_state]][0]

    current_state = states[-1]
    if current_state == bull_state: return "BULL"
    if current_state == bear_state: return "BEAR"
    return "SIDEWAYS"

# 2. DYNAMIC FACTOR SCORER
def compute_timed_factor_score(price_df, state):
    """Adjusts factor weights based on the HMM State."""

    # Calculate Base Factors (Ranked 0 to 1)
    # Momentum (126d Return)
    mom = (price_df.iloc[-1] / price_df.iloc[-126] - 1).rank(pct=True)
    # Low Vol (Inverse 60d Vol)
    vol = price_df.pct_change().iloc[-60:].std()
    low_vol = (1 / vol).rank(pct=True)
    # Quality (Simplified proxy: Inverse 1-yr drawdown)
    dd = (price_df / price_df.rolling(252).max() - 1).iloc[-1].rank(pct=True)

    # DYNAMIC WEIGHTING MATRIX
    if state == "BULL":
        # Aggressive: High Momentum focus
        score = (0.7 * mom) + (0.3 * low_vol)
    elif state == "SIDEWAYS":
        # Neutral: Balance Value and Quality
        score = (0.5 * dd) + (0.5 * low_vol)
    elif state == "BEAR":
        # Defensive: 100% Low Volatility / Safety
        score = (1.0 * low_vol)

    return score.sort_values(ascending=False)

# ── EXECUTION ─────────────────────────────────────────────────────────────
# 1. Load Data
prices = pd.read_csv('Nifty500_10Yr_Historical.csv', index_col='Date', parse_dates=True)
nifty = prices.mean(axis=1)

# 2. Get Current Market Regime
current_regime = get_hmm_state(nifty)
print(f"Current HMM Regime: {current_regime}")

# 3. Generate Timed Factor Picks
top_picks = compute_timed_factor_score(prices.tail(252), current_regime)
print(f"\nTop 10 Timed Factor Picks for {current_regime} Market:")
print(top_picks.head(10))