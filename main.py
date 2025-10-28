import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import gymnasium as gym
from gymnasium import spaces

def fetch_prices(tickers: List[str], start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Download close prices for tickers using yfinance, drop rows with NaNs.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    # If single ticker, keep DataFrame shape
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    # If MultiIndex columns (when single ticker string caused multi), flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna(how="all")
    df = df.ffill().dropna()  # forward-fill and drop any remaining NaNs
    return df


def prepare_features(prices: pd.DataFrame, window: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create features DataFrame and returns DataFrame of simple returns.
    Features per asset: ret, ma5, ma10, vol10, vol20, mom
    """
    prices = prices.sort_index()
    returns = prices.pct_change().fillna(0.0)
    ma5 = prices.rolling(5).mean().bfill()
    ma10 = prices.rolling(10).mean().bfill()
    vol10 = returns.rolling(10).std().fillna(0.0)
    vol20 = returns.rolling(20).std().fillna(0.0)
    momentum = prices / ma5 - 1.0

    features = {}
    for t in prices.columns:
        df_t = pd.DataFrame({
            f"{t}_ret": returns[t],
            f"{t}_ma5": ma5[t],
            f"{t}_ma10": ma10[t],
            f"{t}_vol10": vol10[t],
            f"{t}_vol20": vol20[t],
            f"{t}_mom": momentum[t]
        }, index=prices.index)
        features[t] = df_t

    feat_df = pd.concat(features.values(), axis=1)
    feat_df = feat_df.fillna(method="ffill").fillna(0.0)
    return feat_df, returns


def annualized_return(cum_returns: List[float], periods_per_year: int = 252) -> float:
    total_periods = len(cum_returns)
    if total_periods < 2:
        return 0.0
    total_return = cum_returns[-1] / cum_returns[0] - 1.0
    return (1 + total_return) ** (periods_per_year / total_periods) - 1


def annualized_vol(returns: List[float], periods_per_year: int = 252) -> float:
    arr = np.array(returns)
    return float(np.std(arr) * np.sqrt(periods_per_year)) if len(arr) > 1 else 0.0


def sharpe_ratio(returns: List[float], rf: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - rf / periods_per_year
    sr = np.mean(excess) / (np.std(excess) + 1e-9)
    return float(sr * np.sqrt(periods_per_year))


def max_drawdown(cum_values: List[float]) -> float:
    arr = np.array(cum_values)
    if len(arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / (peak + 1e-9)
    return float(np.min(drawdown))


# -------------------------
# Gymnasium Environment
# -------------------------
class PortfolioEnvGymnasium(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 feat_df: pd.DataFrame,
                 returns_df: pd.DataFrame,
                 window_size: int = 20,
                 reward_window: int = 20,
                 transaction_cost: float = 1e-3,
                 reward_scaling: float = 100.0,
                 initial_cash: float = 1.0,
                 seed: int | None = None):
        super().__init__()
        self.features = feat_df
        self.returns = returns_df
        self.dates = self.features.index
        self.T = len(self.dates)
        self.tickers = self.returns.columns.tolist()
        self.n_assets = len(self.tickers)

        self.window_size = int(window_size)
        self.reward_window = int(reward_window)
        self.transaction_cost = float(transaction_cost)
        self.reward_scaling = float(reward_scaling)
        self.initial_cash = float(initial_cash)

        # feature count per asset (assumes same structure)
        example_cols = [c for c in self.features.columns if c.startswith(self.tickers[0] + "_")]
        self.features_per_asset = len(example_cols)
        self.obs_dim = self.window_size * self.n_assets * self.features_per_asset + self.n_assets

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.n_assets,), dtype=np.float32)

        if seed is not None:
            np.random.seed(seed)

        self._reset_state()

    def _reset_state(self):
        # pick a random start index leaving room for reward_window
        if self.T <= self.window_size + self.reward_window + 1:
            self.start_index = self.window_size
        else:
            self.start_index = np.random.randint(self.window_size, self.T - self.reward_window - 1)
        self.current_step = self.start_index
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.cumulative_return = self.initial_cash
        self.portfolio_returns_history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        obs = self._get_observation()
        info = {"date": self.dates[self.current_step], "cumulative_return": self.cumulative_return}
        return obs, info

    def _get_observation(self):
        t = self.current_step
        start = max(0, t - self.window_size + 1)
        window_dates = self.dates[start:t + 1]

        arr = []
        for dt in window_dates:
            row = []
            for tk in self.tickers:
                cols = [f"{tk}_ret", f"{tk}_ma5", f"{tk}_ma10", f"{tk}_vol10", f"{tk}_vol20", f"{tk}_mom"]
                extracted = [self.features.loc[dt, c] for c in cols if c in self.features.columns]
                if len(extracted) < self.features_per_asset:
                    extracted += [0.0] * (self.features_per_asset - len(extracted))
                row.extend(extracted)
            arr.append(row)

        if len(arr) < self.window_size:
            pad_rows = self.window_size - len(arr)
            pad = [[0.0] * (self.n_assets * self.features_per_asset)] * pad_rows
            arr = pad + arr

        feat_window = np.array(arr, dtype=np.float32).flatten()
        prev_w = np.array(self.current_weights, dtype=np.float32)
        obs = np.concatenate([feat_window, prev_w])
        return obs

    def step(self, action):
        exp = np.exp(action - np.max(action))
        weights = exp / (np.sum(exp) + 1e-9)

        t = self.current_step
        terminated = False

        if (t + 1) < self.T:
            returns_t1 = self.returns.iloc[t + 1].values.astype(np.float32)
        else:
            returns_t1 = np.zeros(self.n_assets, dtype=np.float32)
            terminated = True

        portfolio_return = float(np.dot(weights, returns_t1))

        turnover = float(np.sum(np.abs(weights - self.current_weights)))
        tx_cost = self.transaction_cost * turnover

        net_return = portfolio_return - tx_cost
        self.cumulative_return *= (1 + net_return)

        self.portfolio_returns_history.append(net_return)
        if len(self.portfolio_returns_history) > self.reward_window:
            self.portfolio_returns_history.pop(0)

        if len(self.portfolio_returns_history) >= 2:
            reward = np.mean(self.portfolio_returns_history) / (np.std(self.portfolio_returns_history) + 1e-9)
        else:
            reward = net_return

        reward *= self.reward_scaling

        self.current_weights = weights.copy()
        self.current_step += 1

        obs = self._get_observation()
        info = {
            "date": self.dates[min(self.current_step, self.T - 1)],
            "cumulative_return": float(self.cumulative_return),
            "portfolio_return": float(portfolio_return),
            "turnover": float(turnover),
            "tx_cost": float(tx_cost)
        }

        terminated = self.current_step >= self.T
        truncated = False

        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Weights {np.round(self.current_weights,3)} | CumReturn: {self.cumulative_return:.4f}")


def create_env_from_tickers(tickers: List[str], start: str = "2018-01-01", end: str | None = None,
                            window_size: int = 20, reward_window: int = 20,
                            transaction_cost: float = 1e-3, reward_scaling: float = 50.0) -> PortfolioEnvGymnasium:
    prices = fetch_prices(tickers, start=start, end=end)
    feat_df, returns_df = prepare_features(prices, window=window_size)
    env = PortfolioEnvGymnasium(feat_df, returns_df, window_size=window_size, reward_window=reward_window,
                                transaction_cost=transaction_cost, reward_scaling=reward_scaling)
    return env


def train_model(env: gym.Env, total_timesteps: int = 20000, verbose: int = 1, model_path: str | None = None):
    """
    Train PPO on the given env and save to model_path if provided.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=verbose, batch_size=64, n_steps=512, learning_rate=3e-4)
    model.learn(total_timesteps=int(total_timesteps))
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save(model_path)
    return model


def evaluate_verbose(env: gym.Env, model, episodes: int = 1, deterministic: bool = True):
    results = []
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_info = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_info.append(info)
        results.append(ep_info)
    return results


def compute_performance_metrics(info_ep: list):
    cum = [x["cumulative_return"] for x in info_ep]
    per_step_returns = [x.get("portfolio_return", 0.0) for x in info_ep]
    ann_ret = annualized_return(cum)
    ann_vol = annualized_vol(per_step_returns)
    sr = sharpe_ratio(per_step_returns)
    mdd = max_drawdown(cum)
    return {"annualized_return": ann_ret, "annualized_vol": ann_vol, "sharpe": sr, "max_drawdown": mdd}
