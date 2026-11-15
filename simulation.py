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


def load_backtest_prices(
    tickers: list[str],
    cutoff_date: date,
    lookback_years: int = 3,
) -> pd.DataFrame:
    """Download adjusted closing prices for a historical train/test split.

    The data download may include later dates, but `run_historical_backtest`
    explicitly limits parameter estimation to observations on/before cutoff.
    """
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Enter at least one unique stock ticker.")
    if lookback_years < 1:
        raise ValueError("Historical lookback must be at least one year.")
    cutoff = pd.Timestamp(cutoff_date).normalize()
    if cutoff >= pd.Timestamp(date.today()):
        raise ValueError("Backtest cutoff must be in the past.")

    # Extra history allows the first return inside the requested window.
    start_date = (cutoff - pd.DateOffset(years=lookback_years) - pd.Timedelta(days=12)).date()
    end_date = date.today() + timedelta(days=1)  # Yahoo end is exclusive.
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise ValueError("No historical data returned. Check your tickers or date.")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    missing = [ticker for ticker in tickers if ticker not in close or close[ticker].dropna().empty]
    if missing:
        raise ValueError("Missing historical prices for: " + ", ".join(missing))

    # Yahoo may reorder ticker columns; always align to the inputs.
    close = close.loc[:, tickers].sort_index().dropna(how="any")
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close[~close.index.duplicated(keep="last")]
    if close.empty:
        raise ValueError("No overlapping price dates for your selected tickers.")
    return close


def run_historical_backtest(
    prices: pd.DataFrame,
    cutoff_date: date,
    lookback_years: int,
    weights: list[float],
    initial_portfolio: float = 10_000,
    num_simulations: int = 1_000,
    horizon_days: int = 252,
    risk_free_rate: float = 0.04,
    seed: int = 42,
) -> dict:
    """Out-of-sample portfolio backtest without future data in calibration.

    Training: adjusted daily-close returns from the selected lookback window
    ending at the last trading session ON OR BEFORE cutoff_date.
    Testing: the NEXT horizon_days available trading sessions; uses the same
    constant weights/daily rebalancing assumptions as `run_simulation`.
    """
    if horizon_days < 1:
        raise ValueError("Backtest horizon must be positive.")
    if lookback_years < 1:
        raise ValueError("Lookback must be at least one year.")
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Price history is empty.")
    cutoff = pd.Timestamp(cutoff_date).normalize()
    training_start = cutoff - pd.DateOffset(years=lookback_years)
    train_prices = prices.loc[(prices.index >= training_start) & (prices.index <= cutoff)]
    if len(train_prices) < 62:
        raise ValueError("Not enough pre-cutoff observations. Choose an earlier cutoff or longer-lived tickers.")
    train_returns = train_prices.pct_change().dropna(how="any")
    if len(train_returns) < 60:
        raise ValueError("At least 60 historical return observations are required before the cutoff.")
    if not np.isfinite(train_returns.to_numpy()).all():
        raise ValueError("Training returns contain invalid values.")

    last_train_date = train_prices.index[-1]
    test_prices = prices.loc[prices.index > cutoff].head(horizon_days)
    if len(test_prices) < horizon_days:
        raise ValueError(
            f"Only {len(test_prices)} of {horizon_days} test trading days are available "
            "after the cutoff. Choose an earlier cutoff date or a shorter test horizon."
        )
    train_last_prices = train_prices.iloc[-1]
    # Compare each subsequent session to the prior close; first day's return
    # uses the last training-day close, NOT the first future-day close.
    comparison = pd.concat([train_last_prices.to_frame().T, test_prices], axis=0)
    actual_stock_returns = comparison.pct_change().iloc[1:]
    if not np.isfinite(actual_stock_returns.to_numpy()).all():
        raise ValueError("Test-period returns contain invalid values.")

    w = np.asarray(weights, dtype=float)
    if len(w) != len(prices.columns) or np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError("Portfolio allocations must match tickers and total 100%.")

    model = run_simulation(
        mean_returns=train_returns.mean(),
        covariance=train_returns.cov(),
        weights=w.tolist(),
        initial_portfolio=initial_portfolio,
        num_simulations=num_simulations,
        num_days=horizon_days,
        risk_free_rate=risk_free_rate,
        seed=seed,
    )
    realized_daily_returns = actual_stock_returns.to_numpy() @ w
    actual_values = initial_portfolio * np.cumprod(1.0 + realized_daily_returns)
    actual_end = float(actual_values[-1])
    final_distribution = model["final_values"]
    p5 = float(np.percentile(final_distribution, 5))
    p95 = float(np.percentile(final_distribution, 95))

    return {
        "model": model,
        "training_start": train_prices.index[0],
        "training_end": last_train_date,
        "test_dates": test_prices.index,
        "training_observations": len(train_returns),
        "actual_path": actual_values,
        "actual_ending_value": actual_end,
        "actual_total_return": actual_end / initial_portfolio - 1.0,
        "predicted_median_ending_value": float(model["statistics"]["median_ending_value"]),
        "inside_90_percent_interval": bool(p5 <= actual_end <= p95),
        "actual_percentile_of_simulations": float(np.mean(final_distribution <= actual_end)),
    }
