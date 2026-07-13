"""
NiftyQuantSignals: HMM-Based Factor Timing Engine
================================================
Logic: Dynamically rotates QVMQ factor weights based on
       the Hidden Markov Model state. Includes auto-updater
       and rolling historical backtesting.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

# Import the extracted backtesting module
from backtesting import run_backtest, analyze_performance

# Suppress yfinance and hmmlearn warnings for cleaner terminal output
warnings.filterwarnings("ignore")

HMM_LOOKBACK = 756
TRANSACTION_COST_BPS = 10
OFFICIAL_BENCHMARK_TICKER = "^CRSLDX"

# ── 0. OFFICIAL BENCHMARK HELPER & FETCHER ─────────────────────────────────

def build_official_index_series(prices, official_benchmark_prices):
    return official_benchmark_prices.reindex(prices.index).ffill()

def fetch_official_benchmark(start_date, end_date, ticker=OFFICIAL_BENCHMARK_TICKER):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"[!] No data returned for official benchmark {ticker}.")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                series = data['Adj Close']
            else:
                series = data['Close']
        else:
            series = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)

        return series.dropna()

    except Exception as e:
        print(f"[!] Could not fetch official benchmark {ticker} from Yahoo Finance: {e}")
        return None

# ── 1. DATA UPDATER ───────────────────────────────────────────────────────

def update_historical_prices(csv_filepath="Nifty500_10Yr_Historical.csv"):
    print(f"Checking {csv_filepath} for updates...")

    try:
        prices = pd.read_csv(csv_filepath, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {csv_filepath} not found. Please ensure the file exists.")
        return None

    last_date = prices.index[-1]
    start_date = last_date + timedelta(days=1)

    try:
        benchmark = yf.download("^NSEI", period="1d", progress=False)
        if not benchmark.empty:
            if benchmark.index.tz is not None:
                real_latest_date = benchmark.index[-1].tz_localize(None).date()
            else:
                real_latest_date = benchmark.index[-1].date()
        else:
            real_latest_date = datetime.now().date()
    except Exception:
        real_latest_date = datetime.now().date()

    if start_date.date() > real_latest_date:
        print(f"CSV is current up to {last_date.date()}.")
        print("You already have the latest available market data. Skipping download.\n")
        return prices

    tickers = prices.columns.tolist()
    yf_tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]

    print(f"Fetching new data from {start_date.date()} to {real_latest_date}...")

    new_data = yf.download(
        tickers=yf_tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=(real_latest_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        progress=False
    )

    if isinstance(new_data.columns, pd.MultiIndex):
        if 'Adj Close' in new_data.columns.get_level_values(0):
            new_prices = new_data['Adj Close']
        elif 'Close' in new_data.columns.get_level_values(0):
            new_prices = new_data['Close']
        else:
            new_prices = new_data
    else:
        new_prices = new_data

    ticker_mapping = dict(zip(yf_tickers, tickers))
    new_prices = new_prices.rename(columns=ticker_mapping)
    new_prices = new_prices.reindex(columns=tickers)

    if new_prices.index.tz is not None:
        new_prices.index = new_prices.index.tz_localize(None)

    new_prices = new_prices.dropna(how='all')

    if new_prices.empty:
        print("No new trading days found to append.\n")
        return prices

    no_data_tickers = new_prices.columns[new_prices.isna().all()].tolist()

    if no_data_tickers:
        print(f"\n[!] Warning: No data returned for {len(no_data_tickers)} tickers in this update window.")
        print(f"    These will be kept blank (NaN) in the database.")
        print(f"    Missing Tickers: {', '.join(no_data_tickers)}\n")

    updated_prices = pd.concat([prices, new_prices])
    updated_prices = updated_prices[~updated_prices.index.duplicated(keep='last')]
    updated_prices = updated_prices.sort_index()

    updated_prices.to_csv(csv_filepath)
    print(f"Successfully appended {len(new_prices)} actual trading days to {csv_filepath}.\n")

    return updated_prices

# ── 2. HMM REGIME CLASSIFIER ──────────────────────────────────────────────

def get_hmm_state(price_series):
    ret = np.log(price_series / price_series.shift(1)).fillna(0)
    vol = ret.rolling(5).std().fillna(0)

    obs_df = pd.concat([ret, vol], axis=1).dropna()
    obs = obs_df.values

    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(obs)

    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=42)
    model.fit(obs_scaled)
    states = model.predict(obs_scaled)

    means = model.means_[:, 0]
    sorted_states = np.argsort(means)

    bear_state = sorted_states[0]
    sideways_state = sorted_states[1]
    bull_state = sorted_states[2]

    current_state = states[-1]

    if current_state == bull_state: return "BULL"
    if current_state == bear_state: return "BEAR"
    return "SIDEWAYS"

# ── 3. DYNAMIC FACTOR SCORER ──────────────────────────────────────────────

def compute_timed_factor_score(price_df, state):
    mom = (price_df.iloc[-1] / price_df.iloc[-126] - 1).rank(pct=True)
    vol = price_df.pct_change().iloc[-60:].std()
    low_vol = vol.rank(pct=True, ascending=False)
    dd = (price_df / price_df.rolling(252).max() - 1).iloc[-1].rank(pct=True)

    if state == "BULL":
        score = (0.7 * mom) + (0.3 * low_vol)
    elif state == "SIDEWAYS":
        score = (0.5 * dd) + (0.5 * low_vol)
    elif state == "BEAR":
        score = (1.0 * low_vol)
    else:
        raise ValueError(f"Unrecognized regime state: {state!r}")

    return score.sort_values(ascending=False)

# ── 4. MAIN EXECUTION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    prices = update_historical_prices('Nifty500_10Yr_Historical.csv')

    if prices is not None:
        print(f"\nFetching official benchmark ({OFFICIAL_BENCHMARK_TICKER}) from Yahoo Finance...")
        official_benchmark_prices = fetch_official_benchmark(
            start_date=prices.index[0].strftime('%Y-%m-%d'),
            end_date=(prices.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        if official_benchmark_prices is None or official_benchmark_prices.empty:
            raise SystemExit(
                f"Could not fetch official benchmark {OFFICIAL_BENCHMARK_TICKER} from "
                "Yahoo Finance. Check your network connection and try again."
            )

        nifty_index = build_official_index_series(prices, official_benchmark_prices)

        current_regime = get_hmm_state(nifty_index.tail(HMM_LOOKBACK))
        print(f"\n--- CURRENT MARKET STATUS ---")
        print(f"Current HMM Regime: {current_regime}")

        top_picks = compute_timed_factor_score(prices.tail(252), current_regime)
        print(f"\nTop 10 Timed Factor Picks for {current_regime} Market:")
        print(top_picks.head(10))

        # Run Backtest using Dependency Injection
        port_ret, bench_ret, regimes = run_backtest(
            prices=prices,
            official_benchmark_prices=official_benchmark_prices,
            hmm_classifier_func=get_hmm_state,
            factor_scorer_func=compute_timed_factor_score,
            rebalance_freq=21,
            hmm_lookback=HMM_LOOKBACK,
            transaction_cost_bps=TRANSACTION_COST_BPS
        )

        analyze_performance(port_ret, bench_ret)