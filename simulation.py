"""Monte Carlo portfolio calculations, extracted from the Colab notebook.

Educational model: correlated normally distributed daily arithmetic returns,
constant portfolio weights (daily rebalancing), no transaction costs.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def load_market_data(tickers: list[str], lookback_years: int = 3):
    """Return aligned historical mean daily returns and covariance by ticker.

    Data are adjusted daily closing prices from Yahoo Finance. Missing trading
    dates are dropped jointly so covariance uses an aligned return history.
    """
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Enter at least one unique stock ticker.")

    end_date = date.today() + timedelta(days=1)  # yfinance `end` is exclusive
    start_date = end_date - timedelta(days=365 * lookback_years)

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError("No price history returned. Check the ticker symbols.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:  # One ticker may produce a flat column index
        close = data[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    missing = [t for t in tickers if t not in close or close[t].dropna().empty]
    if missing:
        raise ValueError("Missing market data for: " + ", ".join(missing))

    # Yahoo may alphabetize ticker columns; explicitly restore input order.
    close = close.loc[:, tickers].dropna(how="any")
    returns = close.pct_change().dropna()
    if len(returns) < 60:
        raise ValueError("Too few overlapping historical observations; use older tickers or a longer lookback.")

    return returns.mean(), returns.cov(), len(returns)


def run_simulation(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: list[float],
    initial_portfolio: float = 10_000,
    num_simulations: int = 1_000,
    num_days: int = 252,
    risk_free_rate: float = 0.04,
    seed: int = 42,
):
    """Run the simulation and compute risk statistics.

    Weights are decimals, e.g. [0.3, 0.4, 0.3]; risk_free_rate is annual.
    """
    weights = np.asarray(weights, dtype=float)
    means = mean_returns.to_numpy(dtype=float)
    cov = covariance.loc[mean_returns.index, mean_returns.index].to_numpy(dtype=float)

    if len(weights) != len(means) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Enter one non-negative weight per ticker; weights must sum to 100%.")
    if initial_portfolio <= 0 or num_days <= 0 or num_simulations <= 0:
        raise ValueError("Investment, horizon, and simulation count must be positive.")
    if not np.isfinite(means).all() or not np.isfinite(cov).all():
        raise ValueError("Historical statistics contain non-finite values.")

    annual_return = float(weights @ means * TRADING_DAYS_PER_YEAR)
    annual_volatility = float(np.sqrt(max(0.0, weights @ cov @ weights)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else float("nan")

    rng = np.random.default_rng(seed)
    paths = np.empty((num_days, num_simulations), dtype=float)
    for simulation in range(num_simulations):
        daily_stock_returns = rng.multivariate_normal(means, cov, size=num_days)
        daily_portfolio_returns = daily_stock_returns @ weights
        paths[:, simulation] = initial_portfolio * np.cumprod(1 + daily_portfolio_returns)

    finals = paths[-1, :]
    p05, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
    fifth_final = float(np.percentile(finals, 5))
    worst_final_values = finals[finals <= fifth_final]

    # Losses are measured relative to starting value; floor at zero to avoid
    # reporting a negative VaR when even the fifth percentile shows a gain.
    var_95 = max(0.0, initial_portfolio - fifth_final)
    cvar_95 = max(0.0, initial_portfolio - float(np.mean(worst_final_values)))

    with_initial = np.vstack((np.full((1, num_simulations), initial_portfolio), paths))
    running_peaks = np.maximum.accumulate(with_initial, axis=0)
    drawdown_magnitudes = -((with_initial - running_peaks) / running_peaks).min(axis=0)

    return {
        "paths": paths,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "final_values": finals,
        "statistics": {
            "expected_annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": float(sharpe),
            "mean_ending_value": float(np.mean(finals)),
            "median_ending_value": float(np.median(finals)),
            "fifth_percentile_ending_value": fifth_final,
            "ninety_fifth_percentile_ending_value": float(np.percentile(finals, 95)),
            "probability_of_loss": float(np.mean(finals < initial_portfolio)),
            "var_95_dollars": var_95,
            "cvar_95_dollars": cvar_95,
            "average_max_drawdown": float(np.mean(drawdown_magnitudes)),
            "median_max_drawdown": float(np.median(drawdown_magnitudes)),
            "drawdown_95": float(np.percentile(drawdown_magnitudes, 95)),
        },
    }
