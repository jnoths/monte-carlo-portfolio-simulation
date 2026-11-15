"""Interactive Streamlit dashboard for the Monte Carlo portfolio model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulation import load_market_data, run_simulation

st.set_page_config(page_title="Monte Carlo Portfolio Simulator", page_icon="📈", layout="wide")
st.title("Monte Carlo Portfolio Simulator")
st.write("Explore how changing stocks, allocations, and simulation settings affects modeled portfolio outcomes.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...], lookback_years: int):
    return load_market_data(list(tickers), lookback_years)


# Inputs are deliberately outside st.form. Changing a widget reruns the app,
# so the simulation and charts below respond to the latest input values.
with st.sidebar:
    st.header("Simulation settings")
    st.caption("Charts recalculate automatically when you change a setting.")
    ticker_text = st.text_input("Stock tickers (comma separated)", value="AAPL, MSFT, GOOG")
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    valid_tickers = 1 <= len(tickers) <= 6 and len(set(tickers)) == len(tickers)

    weights_percent = []
    if valid_tickers:
        st.caption("Adjust relative allocations below. The app normalizes them to 100%.")
        raw_allocations = []
        for ticker in tickers:
            raw_allocations.append(
                st.slider(
                    f"{ticker} allocation preference",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    key=f"relative_allocation_{ticker}",
                )
            )
        allocation_sum = sum(raw_allocations)
        if allocation_sum > 0:
            weights_percent = [100.0 * x / allocation_sum for x in raw_allocations]
            st.caption(
                "Actual portfolio weights: "
                + ", ".join(f"{t} {w:.1f}%" for t, w in zip(tickers, weights_percent))
            )
        else:
            st.warning("Move at least one allocation slider above zero.")
    else:
        st.warning("Please enter 1–6 different stock ticker symbols.")

    initial_portfolio = st.number_input(
        "Initial investment ($)", min_value=100.0, value=10000.0, step=500.0
    )
    history_years = st.selectbox("Historical lookback (years)", [1, 3, 5], index=1)
    forecast_years = st.selectbox("Forecast horizon (years)", [1, 2, 3], index=0)
    num_simulations = st.selectbox("Number of simulations", [100, 500, 1000, 2500], index=2)
    risk_free_percent = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=25.0, value=4.0, step=0.5
    )

if not valid_tickers or not weights_percent:
    st.info("Enter valid tickers and nonzero allocations to display the graphs.")
    st.stop()

try:
    # Historical prices are cached for an hour, so changing only weights or
    # investment size does not repeatedly call Yahoo Finance.
    with st.spinner("Updating Monte Carlo simulation..."):
        mean_returns, covariance, observations = cached_market_data(tuple(tickers), history_years)
        result = run_simulation(
            mean_returns=mean_returns,
            covariance=covariance,
            weights=[w / 100 for w in weights_percent],
            initial_portfolio=initial_portfolio,
            num_simulations=num_simulations,
            num_days=forecast_years * 252,
            risk_free_rate=risk_free_percent / 100,
            seed=42,
        )
except (ValueError, KeyError, OSError, RuntimeError) as exc:
    st.error(f"Could not update the simulation: {exc}")
    st.stop()

run = {
    "result": result,
    "tickers": tickers,
    "weights": weights_percent,
    "initial": initial_portfolio,
    "years": forecast_years,
    "sims": num_simulations,
    "observations": observations,
}
result = run["result"]
stats = result["statistics"]

st.subheader("Portfolio overview")
st.caption(
    f"{', '.join(f'{t}: {w:.2f}%' for t, w in zip(run['tickers'], run['weights']))} · "
    f"{run['sims']:,} simulations · {run['years']} forecast year(s) · "
    f"{run['observations']:,} historical daily returns"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average ending value", f"${stats['mean_ending_value']:,.0f}")
col2.metric("Probability of loss", f"{stats['probability_of_loss']:.1%}")
col3.metric("95% VaR", f"${stats['var_95_dollars']:,.0f}")
col4.metric("95% expected shortfall", f"${stats['cvar_95_dollars']:,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Simulated portfolio paths")
    fig, ax = plt.subplots(figsize=(8, 5))
    days = np.arange(1, result["paths"].shape[0] + 1)
    # Plot a sample of 200 paths for readability; all paths enter risk calculations.
    ax.plot(days, result["paths"][:, :200], alpha=0.06, linewidth=0.7)
    ax.plot(days, result["p05"], linewidth=2, label="5th percentile")
    ax.plot(days, result["p50"], linewidth=2, label="Median")
    ax.plot(days, result["p95"], linewidth=2, label="95th percentile")
    ax.axhline(run["initial"], linestyle="--", linewidth=1.2, label="Initial investment")
    ax.set(xlabel="Trading days", ylabel="Portfolio value ($)")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with right:
    st.subheader("Distribution of ending values")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result["final_values"], bins=40, alpha=0.8)
    ax.axvline(run["initial"], linestyle="--", label="Initial investment")
    ax.axvline(stats["fifth_percentile_ending_value"], linestyle="--", label="5th percentile")
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

final_values_csv = pd.DataFrame({"simulation": np.arange(1, run["sims"] + 1), "ending_value": result["final_values"]}).to_csv(index=False)
st.download_button("Download simulated ending values (CSV)", final_values_csv, file_name="monte_carlo_ending_values.csv", mime="text/csv")

st.caption(
    "Model assumptions: independent multivariate-normal daily stock returns, constant portfolio weights "
    "(as if rebalanced daily), historical mean/covariance estimates, no fees or transaction costs. "
    "VaR and expected shortfall measure simulated losses relative to the starting amount over the full forecast horizon. "
    "Results are illustrative, not forecasts or investment advice."
)
