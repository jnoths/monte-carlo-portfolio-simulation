"""Monte Carlo Portfolio Simulator with a no-lookahead historical backtest.

Place this file alongside simulation.py and requirements.txt in your GitHub repo.
Run locally with: streamlit run app.py
"""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    load_backtest_prices,
    load_market_data,
    run_historical_backtest,
    run_simulation,
)

st.set_page_config(page_title="Monte Carlo Portfolio Simulator", page_icon="📈", layout="wide")
st.title("Monte Carlo Portfolio Simulator")
st.write(
    "Explore portfolio risk with simulations and check a past forecast "
    "against what actually happened."
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...], lookback_years: int):
    return load_market_data(list(tickers), lookback_years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtest_prices(tickers: tuple[str, ...], cutoff: date, lookback_years: int):
    return load_backtest_prices(list(tickers), cutoff, lookback_years)


# Shared inputs: both the forecast and the historical backtest use the same
# tickers, weights, investment, and simulation settings. No st.form means
# changes to the allocation textboxes rerun the dashboard automatically.
with st.sidebar:
    st.header("Portfolio settings")
    st.caption("Charts update when you change an input and allocations total 100%.")
    ticker_text = st.text_input("Stock tickers (comma separated)", value="AAPL, MSFT, GOOG")
    tickers = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]
    valid_tickers = 1 <= len(tickers) <= 6 and len(set(tickers)) == len(tickers)

    weights_percent = []
    if valid_tickers:
        st.caption("Enter percentage allocations. They should total 100%.")
        equal_part = round(100.0 / len(tickers), 2)
        for i, ticker in enumerate(tickers):
            default_weight = (
                equal_part if i < len(tickers) - 1
                else round(100.0 - equal_part * (len(tickers) - 1), 2)
            )
            weights_percent.append(
                st.number_input(
                    f"{ticker} allocation (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_weight),
                    step=1.0,
                    format="%.2f",
                    key=f"allocation_{ticker}",
                )
            )
        total_weight = sum(weights_percent)
        if abs(total_weight - 100.0) <= 0.011:
            st.caption(f"Total allocation: {total_weight:.2f}% ✓")
        else:
            st.warning(f"Total allocation: {total_weight:.2f}%. Adjust to 100%.")
    else:
        st.warning("Enter between 1 and 6 different stock tickers.")

    investment = st.number_input(
        "Initial investment ($)", min_value=100.0, value=10_000.0, step=500.0
    )
    history_years = st.selectbox("Historical lookback (years)", [1, 3, 5], index=1)
    forecast_years = st.selectbox("Simulation forecast (years)", [1, 2, 3], index=0)
    num_simulations = st.selectbox("Number of simulations", [100, 500, 1000, 2500], index=2)
    risk_free_percent = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=25.0, value=4.0, step=0.5
    )

if not valid_tickers:
    st.info("Enter 1–6 unique tickers to view the analyses.")
    st.stop()
if abs(sum(weights_percent) - 100.0) > 0.011:
    st.info("Adjust the allocation textboxes to total 100% to run the analyses.")
    st.stop()

# Fix small decimal-rounding differences such as 33.33 + 33.33 + 33.33 = 99.99.
weights = [w / sum(weights_percent) for w in weights_percent]

# The tabs track which one is open, so Streamlit only downloads and simulates
# data for the tab the user is viewing. This avoids unnecessary API requests.
forecast_tab, backtest_tab = st.tabs(
    ["📈 Monte Carlo Forecast", "🧪 Historical Backtest"], on_change="rerun"
)

if forecast_tab.open:
    with forecast_tab:
        st.subheader("Monte Carlo forecast")
        with st.spinner("Updating Monte Carlo simulation..."):
            try:
                mean_returns, covariance, observations = cached_market_data(
                    tuple(tickers), history_years
                )
                result = run_simulation(
                    mean_returns=mean_returns,
                    covariance=covariance,
                    weights=weights,
                    initial_portfolio=investment,
                    num_simulations=num_simulations,
                    num_days=forecast_years * 252,
                    risk_free_rate=risk_free_percent / 100,
                    seed=42,
                )
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                st.error(f"Could not update the simulation: {exc}")
            else:
                stats = result["statistics"]
                st.caption(
                    f"{', '.join(f'{t}: {w:.2f}%' for t, w in zip(tickers, weights_percent))} "
                    f"· {num_simulations:,} simulations · {forecast_years} forecast year(s) "
                    f"· {observations:,} historical daily returns"
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Average ending value", f"${stats['mean_ending_value']:,.0f}")
                m2.metric("Probability of loss", f"{stats['probability_of_loss']:.1%}")
                m3.metric("95% VaR", f"${stats['var_95_dollars']:,.0f}")
                m4.metric("95% expected shortfall", f"${stats['cvar_95_dollars']:,.0f}")

                left, right = st.columns(2)
                with left:
                    st.subheader("Simulated portfolio paths")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    days = np.arange(1, result["paths"].shape[0] + 1)
                    ax.plot(days, result["paths"][:, :200], alpha=0.06, linewidth=0.7)
                    ax.plot(days, result["p05"], linewidth=2, label="5th percentile")
                    ax.plot(days, result["p50"], linewidth=2, label="Median")
                    ax.plot(days, result["p95"], linewidth=2, label="95th percentile")
                    ax.axhline(investment, linestyle="--", linewidth=1.2, label="Initial investment")
                    ax.set(xlabel="Trading days", ylabel="Portfolio value ($)")
                    ax.legend(loc="upper left", fontsize="small")
                    ax.grid(alpha=0.2)
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                with right:
                    st.subheader("Distribution of ending values")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(result["final_values"], bins=40, alpha=0.8)
                    ax.axvline(investment, linestyle="--", label="Initial investment")
                    ax.axvline(
                        stats["fifth_percentile_ending_value"],
                        linestyle="--",
                        label="5th percentile",
                    )
                    ax.set(xlabel="Ending portfolio value ($)", ylabel="Number of simulations")
                    ax.legend(fontsize="small")
                    ax.grid(alpha=0.2)
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                st.subheader("Additional risk statistics")
                summary = pd.DataFrame(
                    [
                        ("Expected annual return (historical estimate)", f"{stats['expected_annual_return']:.2%}"),
                        ("Annualized volatility", f"{stats['annual_volatility']:.2%}"),
                        ("Sharpe ratio", f"{stats['sharpe_ratio']:.2f}"),
                        ("Median ending value", f"${stats['median_ending_value']:,.2f}"),
                        ("5th percentile ending value", f"${stats['fifth_percentile_ending_value']:,.2f}"),
                        ("95th percentile ending value", f"${stats['ninety_fifth_percentile_ending_value']:,.2f}"),
                        ("Average maximum drawdown", f"{stats['average_max_drawdown']:.2%}"),
                        ("Median maximum drawdown", f"{stats['median_max_drawdown']:.2%}"),
                        ("95th percentile maximum drawdown", f"{stats['drawdown_95']:.2%}"),
                    ],
                    columns=["Measure", "Value"],
                )
                st.dataframe(summary, hide_index=True, use_container_width=True)
                final_csv = pd.DataFrame(
                    {"simulation": np.arange(1, num_simulations + 1), "ending_value": result["final_values"]}
                ).to_csv(index=False)
                st.download_button(
                    "Download simulated ending values (CSV)",
                    final_csv,
                    file_name="monte_carlo_ending_values.csv",
                    mime="text/csv",
                )

if backtest_tab.open:
    with backtest_tab:
        st.subheader("Historical out-of-sample backtest")
        st.write(
            "Choose a past cutoff date. The model learns only from prices on or before that date, "
            "then simulates the following period and compares it with actual daily portfolio values."
        )

        # About 14 months in the past gives enough completed sessions for a
        # one-year (252 trading-day) test by default.
        default_cutoff = date.today() - timedelta(days=420)
        controls_left, controls_right = st.columns(2)
        with controls_left:
            cutoff_date = st.date_input(
                "Last date the model is allowed to learn from",
                value=default_cutoff,
                min_value=date(2010, 1, 1),
                max_value=date.today() - timedelta(days=1),
                help="If you select a weekend/holiday, the prior available session is used.",
            )
        with controls_right:
            horizon_days = st.selectbox(
                "Backtest horizon",
                options=[126, 252],
                format_func=lambda v: "About 6 months (126 trading days)" if v == 126 else "About 1 year (252 trading days)",
                index=1,
            )

        if not cutoff_date:
            st.warning("Select a backtest cutoff date.")
        else:
            try:
                with st.spinner("Downloading past prices and calculating the backtest..."):
                    price_history = cached_backtest_prices(
                        tuple(tickers), cutoff_date, history_years
                    )
                    bt = run_historical_backtest(
                        prices=price_history,
                        cutoff_date=cutoff_date,
                        lookback_years=history_years,
                        weights=weights,
                        initial_portfolio=investment,
                        num_simulations=num_simulations,
                        horizon_days=horizon_days,
                        risk_free_rate=risk_free_percent / 100,
                        seed=42,
                    )
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                st.error(f"Could not complete the backtest: {exc}")
            else:
                model = bt["model"]
                stats = model["statistics"]
                st.caption(
                    f"Training: {bt['training_start']:%b %d, %Y} – {bt['training_end']:%b %d, %Y} "
                    f"({bt['training_observations']} daily returns) · "
                    f"Testing: {bt['test_dates'][0]:%b %d, %Y} – {bt['test_dates'][-1]:%b %d, %Y} "
                    f"({horizon_days} trading days)"
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Actual ending value", f"${bt['actual_ending_value']:,.0f}")
                m2.metric("Actual return", f"{bt['actual_total_return']:+.1%}")
                m3.metric("Predicted median ending", f"${bt['predicted_median_ending_value']:,.0f}")
                m4.metric(
                    "Actual inside 5th–95th range?",
                    "Yes" if bt["inside_90_percent_interval"] else "No",
                )

                st.subheader("Predicted range vs. actual portfolio performance")
                # Include the common starting value to make it clear that
                # forecast and realized paths begin at the same moment.
                chart_dates = pd.DatetimeIndex([bt["training_end"]]).append(bt["test_dates"])
                p05 = np.r_[investment, model["p05"]]
                median = np.r_[investment, model["p50"]]
                p95 = np.r_[investment, model["p95"]]
                actual = np.r_[investment, bt["actual_path"]]
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.fill_between(
                    chart_dates,
                    p05,
                    p95,
                    alpha=0.22,
                    label="Simulated 5th–95th percentile range",
                )
                ax.plot(chart_dates, median, linewidth=2, label="Simulated median")
                ax.plot(chart_dates, actual, linewidth=2, label="Actual portfolio path")
                ax.axhline(investment, linestyle="--", alpha=0.5, label="Initial investment")
                ax.set(xlabel="Date", ylabel="Portfolio value ($)")
                ax.grid(alpha=0.2)
                ax.legend(fontsize="small")
                fig.autofmt_xdate()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                st.subheader("End-of-period comparison")
                comparison = pd.DataFrame(
                    [
                        ("Actual ending value", f"${bt['actual_ending_value']:,.2f}"),
                        ("Predicted 5th percentile", f"${stats['fifth_percentile_ending_value']:,.2f}"),
                        ("Predicted median", f"${stats['median_ending_value']:,.2f}"),
                        ("Predicted 95th percentile", f"${stats['ninety_fifth_percentile_ending_value']:,.2f}"),
                        ("Actual outcome percentile among simulated endpoints", f"{bt['actual_percentile_of_simulations']:.1%}"),
                    ],
                    columns=["Measure", "Value"],
                )
                st.dataframe(comparison, hide_index=True, use_container_width=True)
                st.info(
                    "This is one historical test, not proof of forecast accuracy. "
                    "The displayed 5th–95th range is a **pointwise** percentile band; "
                    "the 90% interpretation applies to the distribution of ending values, "
                    "not the chance that the entire actual path stays inside it. "
                    "Try other cutoff dates to see how the model performs in different markets."
                )

                comparison_csv = pd.DataFrame(
                    {
                        "date": chart_dates.strftime("%Y-%m-%d"),
                        "actual_value": actual,
                        "forecast_p05": p05,
                        "forecast_median": median,
                        "forecast_p95": p95,
                    }
                ).to_csv(index=False)
                st.download_button(
                    "Download backtest comparison (CSV)",
                    comparison_csv,
                    file_name="monte_carlo_backtest.csv",
                    mime="text/csv",
                )

st.caption(
    "Educational model assumptions: independent multivariate-normal daily stock returns, "
    "fixed weights (equivalent to daily rebalancing), historical mean/covariance estimates, "
    "no transaction costs or fees. Historical adjusted prices are sourced from Yahoo Finance. "
    "Results are illustrative, not guaranteed investment forecasts or financial advice."
)
