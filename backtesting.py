"""
NiftyQuantSignals: Backtesting Engine
================================================
Handles historical rolling simulations, transaction cost
accounting, and comprehensive performance metric visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(
    prices,
    official_benchmark_prices,
    hmm_classifier_func,
    factor_scorer_func,
    rebalance_freq=21,
    hmm_lookback=756,
    transaction_cost_bps=10
):
    """
    Simulates a rolling backtest of the HMM Factor strategy.
    """
    if official_benchmark_prices is None or official_benchmark_prices.empty:
        raise ValueError(
            "official_benchmark_prices is required. Check network access to "
            "Yahoo Finance and retry -- there is no offline fallback proxy."
        )

    print("\nInitializing Backtest Engine...")

    daily_returns = prices.pct_change().shift(-1).clip(lower=-0.20, upper=0.20)

    nifty_index = official_benchmark_prices.reindex(prices.index).ffill()
    benchmark_returns = nifty_index.pct_change().shift(-1)

    portfolio_returns = pd.Series(index=prices.index, dtype=float)
    historical_regimes = []

    previous_holdings = set()
    start_idx = hmm_lookback

    print(f"Total Trading Days: {len(prices)}")
    print(f"Backtest Start Date: {prices.index[start_idx].date()}")
    print(f"Transaction Cost Assumption: {transaction_cost_bps} bps one-way, charged on turnover")
    print("-" * 50)

    for i in range(start_idx, len(prices), rebalance_freq):
        current_date = prices.index[i]

        history_prices = prices.iloc[:i+1]
        history_nifty = nifty_index.iloc[:i+1]

        try:
            regime = hmm_classifier_func(history_nifty.tail(hmm_lookback))
        except Exception:
            regime = "SIDEWAYS"

        historical_regimes.append((current_date, regime))

        scores = factor_scorer_func(history_prices.tail(252), regime)
        top_10 = scores.head(10).index.tolist()
        current_holdings = set(top_10)

        if previous_holdings:
            names_changed = len(current_holdings.symmetric_difference(previous_holdings))
            turnover = names_changed / (2 * len(current_holdings))
        else:
            turnover = 1.0

        cost_drag = turnover * (transaction_cost_bps / 10000)
        previous_holdings = current_holdings

        end_idx = min(i + rebalance_freq, len(prices))
        for j in range(i, end_idx):
            day_date = prices.index[j]
            day_ret = daily_returns.loc[day_date, top_10].mean()

            if j == i:
                day_ret -= cost_drag

            portfolio_returns.loc[day_date] = day_ret

    portfolio_returns = portfolio_returns.dropna()
    benchmark_returns = benchmark_returns.loc[portfolio_returns.index].dropna()

    return portfolio_returns, benchmark_returns, historical_regimes


def analyze_performance(port_returns, bench_returns, risk_free_rate=0.06):
    """
    Calculates CAGR, Volatility, Sharpe Ratio, Calmar Ratio, Max Drawdown,
    and plots the equity curve.
    risk_free_rate default is 6% (0.06), typical for Indian government bonds.
    """

    port_cum = (1 + port_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    # 1. Compound Annual Growth Rate (CAGR)
    years = len(port_returns) / 252
    port_cagr = (port_cum.iloc[-1]**(1/years) - 1)
    bench_cagr = (bench_cum.iloc[-1]**(1/years) - 1)

    # 2. Annualized Volatility
    port_vol = port_returns.std() * np.sqrt(252)
    bench_vol = bench_returns.std() * np.sqrt(252)

    # 3. Sharpe Ratio
    port_sharpe = (port_cagr - risk_free_rate) / port_vol if port_vol != 0 else 0
    bench_sharpe = (bench_cagr - risk_free_rate) / bench_vol if bench_vol != 0 else 0

    # 4. Max Drawdown Calculator
    def get_max_drawdown(cum_returns):
        rolling_max = cum_returns.cummax()
        drawdown = cum_returns / rolling_max - 1
        return drawdown.min()

    port_mdd = get_max_drawdown(port_cum)
    bench_mdd = get_max_drawdown(bench_cum)

    # 5. Calmar Ratio
    port_calmar = port_cagr / abs(port_mdd) if port_mdd != 0 else 0
    bench_calmar = bench_cagr / abs(bench_mdd) if bench_mdd != 0 else 0

    # Performance Tear Sheet
    print("\n" + "="*55)
    print("BACKTEST PERFORMANCE (HMM FACTOR ENGINE)")
    print("="*55)
    print(f"{'Metric':<22} | {'Strategy':<12} | {'Benchmark':<12}")
    print("-" * 55)
    print(f"{'Total Return':<22} | {(port_cum.iloc[-1]-1)*100:>11.2f}% | {(bench_cum.iloc[-1]-1)*100:>11.2f}%")
    print(f"{'CAGR (Annualized)':<22} | {port_cagr*100:>11.2f}% | {bench_cagr*100:>11.2f}%")
    print(f"{'Max Drawdown':<22} | {port_mdd*100:>11.2f}% | {bench_mdd*100:>11.2f}%")
    print(f"{'Ann. Volatility':<22} | {port_vol*100:>11.2f}% | {bench_vol*100:>11.2f}%")
    print(f"{'Sharpe Ratio (Rf=6%)':<22} | {port_sharpe:>11.2f}  | {bench_sharpe:>11.2f} ")
    print(f"{'Calmar Ratio':<22} | {port_calmar:>11.2f}  | {bench_calmar:>11.2f} ")
    print("="*55)
    print("Note: Benchmark is the official Nifty 500 (^CRSLDX), PRICE RETURN")
    print("only (no dividends) -- expect it to sit a bit below NSE's quoted")
    print("Total Return Index figure, roughly by the dividend yield.")

    # Chart Generation
    plt.figure(figsize=(12, 6))
    plt.plot(port_cum, label='HMM Factor Strategy', color='blue', linewidth=2)
    plt.plot(bench_cum, label='Benchmark (Nifty 500 Official, ^CRSLDX)', color='gray', linestyle='--')
    plt.title('HMM Dynamic Factor Timing vs Benchmark')
    plt.ylabel('Cumulative Growth (1x = Starting Capital)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()