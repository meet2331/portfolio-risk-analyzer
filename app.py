import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Risk Analyser", layout="wide")
st.title("Portfolio Risk & Stress Test Analyser")

# --- Portfolio input ---
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.multiselect(
    "Select assets",
    ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "DB", "CBK.DE", "BAYN.DE"],
    default=["JPM", "GS", "DB"]
)
weights_input = st.sidebar.text_input("Weights (comma-separated, must sum to 1)", "0.4,0.3,0.3")
portfolio_value = st.sidebar.number_input("Portfolio Value (€)", value=100000)
confidence = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95)

if not tickers:
    st.warning("Please select at least one asset.")
    st.stop()

weights = np.array([float(w) for w in weights_input.split(",")])
if abs(weights.sum() - 1.0) > 0.01:
    st.error("Weights must sum to 1.")
    st.stop()

# --- Fetch data ---
@st.cache_data
def load_data(tickers):
    raw = yf.download(tickers, start="2005-01-01", end="2024-12-31")["Close"]
    return raw.dropna()

prices = load_data(tickers)
returns = prices.pct_change().dropna()

# --- Portfolio returns ---
port_returns = returns.dot(weights)

# --- VaR and Expected Shortfall ---
var = np.percentile(port_returns, (1 - confidence) * 100)
es = port_returns[port_returns <= var].mean()

var_eur = abs(var) * portfolio_value
es_eur = abs(es) * portfolio_value

col1, col2, col3 = st.columns(3)
col1.metric("Daily VaR", f"€{var_eur:,.0f}", f"at {int(confidence*100)}% confidence")
col2.metric("Expected Shortfall", f"€{es_eur:,.0f}", "average loss beyond VaR")
col3.metric("Portfolio Value", f"€{portfolio_value:,.0f}")

# --- Return distribution ---
st.subheader("Return Distribution")
fig = px.histogram(port_returns, nbins=100, title="Daily Portfolio Returns")
fig.add_vline(x=var, line_dash="dash", line_color="red", annotation_text=f"VaR ({int(confidence*100)}%)")
fig.add_vline(x=es, line_dash="dot", line_color="orange", annotation_text="Expected Shortfall")
st.plotly_chart(fig, use_container_width=True)

# --- Stress scenarios ---
st.subheader("Stress Test Scenarios")

scenarios = {
    "2008 Financial Crisis (Sep–Nov 2008)": ("2008-09-01", "2008-11-30"),
    "COVID Crash (Feb–Mar 2020)": ("2020-02-01", "2020-03-31"),
    "Dot-com Bust (2001)": ("2001-01-01", "2001-12-31"),
}

results = []
for name, (start, end) in scenarios.items():
    mask = (port_returns.index >= start) & (port_returns.index <= end)
    scenario_returns = port_returns[mask]
    if len(scenario_returns) > 0:
        cumulative_loss = (1 + scenario_returns).prod() - 1
        max_daily_loss = scenario_returns.min()
        results.append({
            "Scenario": name,
            "Cumulative Loss": f"{cumulative_loss*100:.1f}%",
            "Loss in € ": f"€{abs(cumulative_loss)*portfolio_value:,.0f}",
            "Worst Single Day": f"{max_daily_loss*100:.2f}%"
        })

st.dataframe(pd.DataFrame(results), use_container_width=True)

# --- Cumulative portfolio performance ---
st.subheader("Cumulative Portfolio Performance")
cumulative = (1 + port_returns).cumprod()
fig2 = px.line(cumulative, title="Portfolio Growth (normalised to 1)")
for name, (start, end) in scenarios.items():
    fig2.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.1, annotation_text=name.split("(")[0])
st.plotly_chart(fig2, use_container_width=True)
