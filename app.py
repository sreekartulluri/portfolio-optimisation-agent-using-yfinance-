import os
import time
import joblib
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from main import (
    create_env_from_tickers, fetch_prices, prepare_features,
    train_model, evaluate_verbose, compute_performance_metrics
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


st.set_page_config(page_title="RL Portfolio Optimizer", layout="wide")
st.title("📈 RL Portfolio Optimizer")

# --------------------
# Sidebar controls
# --------------------
st.sidebar.header("Data & Model Configuration")
tickers_str = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

window_size = st.sidebar.slider("Window Size (days)", 5, 60, 20)
reward_window = st.sidebar.slider("Reward Window (days)", 5, 60, 20)
timesteps = st.sidebar.number_input("Training timesteps", value=20000, min_value=1000, step=1000)
tx_cost = st.sidebar.number_input("Transaction cost (fraction)", value=0.001, step=0.0001, format="%.4f")
#reward_scaling = st.sidebar.number_input("Reward scaling", value=50.0)

model_name = st.sidebar.text_input("Model filename (save/load)", "ppo_portfolio_upgraded")

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Training is CPU-bound and may take time depending on timesteps.")
st.sidebar.markdown("Use small timesteps for quick experiments.")

# --------------------
# Helpers: caching
# --------------------
@st.cache_data(ttl=60 * 60)
def cached_fetch_prices(tickers, start, end):
    return fetch_prices(tickers, start=start, end=end)


@st.cache_data(ttl=60 * 60)
def cached_prepare_features(prices, window):
    return prepare_features(prices, window=window)


# --------------------
# Main UI
# --------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Data")
    if st.button("Fetch & Preview Data"):
        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            with st.spinner("Downloading price data..."):
                prices = cached_fetch_prices(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                st.success(f"Loaded {prices.shape[0]} rows × {prices.shape[1]} tickers.")
                st.dataframe(prices.tail(10))

                st.subheader("Price chart")
                fig = go.Figure()
                for col in prices.columns:
                    fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col))
                fig.update_layout(height=300, margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model")
    col21, col22 = st.columns([1, 1])
    with col21:
        if st.button("Create Env (preview)"):
            if not tickers:
                st.warning("Enter tickers first.")
            else:
                try:
                    prices = cached_fetch_prices(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    feat_df, returns_df = cached_prepare_features(prices, window_size)
                    st.success("Environment data prepared.")
                    st.write("Feature sample:")
                    st.dataframe(feat_df.iloc[-5:, : min(12, feat_df.shape[1])])
                except Exception as e:
                    st.error(f"Error preparing features: {e}")

    with col22:
        model_path = MODEL_DIR / f"{model_name}.zip"
        if st.button("Train Model"):
            if not tickers:
                st.warning("Enter tickers first.")
            else:
                # Build env
                with st.spinner("Preparing environment..."):
                    env = create_env_from_tickers(tickers, start=start_date.strftime("%Y-%m-%d"),
                                                  end=end_date.strftime("%Y-%m-%d"),
                                                  window_size=window_size, reward_window=reward_window)
                st.info("Starting training... (this will block the UI until finished)")
                # Train
                t0 = time.time()
                model = train_model(env, total_timesteps=timesteps, verbose=1, model_path=str(model_path))
                t1 = time.time()
                st.success(f"Training finished in {int(t1-t0)}s. Model saved to {model_path}")
                st.session_state["model"] = model
                st.session_state["env"] = env

        if st.button("Load Model"):
            if model_path.exists():
                from stable_baselines3 import PPO
                model = PPO.load(str(model_path))
                st.success(f"Loaded model from {model_path}")
                st.session_state["model"] = model
            else:
                st.warning(f"No model found at {model_path}. Train first or change filename.")

# --------------------
# Evaluation & Plots
# --------------------
st.markdown("---")
st.subheader("Evaluate / Visualize")

colA, colB = st.columns([1, 1])

with colA:
    episodes = st.number_input("Evaluation episodes", value=1, min_value=1)
    if st.button("Run Evaluation (deterministic)"):
        if "model" not in st.session_state:
            st.warning("No model loaded. Train or load a model first.")
        else:
            model = st.session_state["model"]
            if "env" not in st.session_state:
                # recreate env for evaluation
                env = create_env_from_tickers(tickers, start=start_date.strftime("%Y-%m-%d"),
                                              end=end_date.strftime("%Y-%m-%d"),
                                              window_size=window_size)
            else:
                env = st.session_state["env"]

            with st.spinner("Running evaluation..."):
                infos_list = evaluate_verbose(env, model, episodes=int(episodes), deterministic=True)

            # show results
            for i, ep in enumerate(infos_list):
                st.write(f"Episode {i+1} — final cumulative return: {ep[-1]['cumulative_return']:.4f}")
                metrics = compute_performance_metrics(ep)
                st.json(metrics)

            # Plot cumulative returns
            fig = go.Figure()
            for i, ep in enumerate(infos_list):
                cum = [x["cumulative_return"] for x in ep]
                fig.add_trace(go.Scatter(y=cum, mode="lines", name=f"ep{i+1}"))
            fig.update_layout(title="Cumulative Return", yaxis_title="Cumulative Return")
            st.plotly_chart(fig, use_container_width=True)

with colB:
    st.subheader("Compare Baseline")
    if st.button("Baseline (Equal Weight)"):
        prices = cached_fetch_prices(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        returns = prices.pct_change().fillna(0.0)
        equal_w = np.repeat(1.0 / len(tickers), len(tickers))
        cum = [1.0]
        for i in range(1, len(returns)):
            r = (returns.iloc[i].values * equal_w).sum()
            cum.append(cum[-1] * (1 + r))
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cum, mode="lines", name="EqualWeight"))
        fig.update_layout(title="Equal Weight Baseline", yaxis_title="Cumulative Return")
        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown("## Tips / Next steps")
st.markdown("""
- Use small timesteps initially (e.g., 2k-10k) to ensure everything works.
- For production or many concurrent users, train on a separate server and only serve evaluation + visualization from the web UI.
- Consider saving training logs and metrics (WandB/MLflow) for better analysis.
""")
