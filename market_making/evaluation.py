from __future__ import annotations

import numpy as np
import pandas as pd

from market_making.simulator import MarketSimulator, SimulatorParams


def run_episode(policy, params: SimulatorParams, seed: int) -> dict:
    sim = MarketSimulator(params=params, seed=seed)
    sim.reset()
    W0 = sim.wealth
    inv_sq, fills, orders, total_reward = 0.0, 0, 0, 0.0
    liquidation_cost, done = 0.0, False
    while not done:
        d_bid, d_ask = policy.act(sim.q, params.q_max, sim.t, params.T)
        _, r, done, info = sim.step(d_bid, d_ask)
        total_reward += r
        inv_sq += sim.q ** 2
        fills += info["f_bid"] + info["f_ask"]
        orders += info["n_buy"] + info["n_sell"]
        liquidation_cost = info["liquidation_cost"]
    return {
        "pnl": sim.wealth - liquidation_cost - W0,
        "rms_inventory": np.sqrt(inv_sq / params.T),
        "fill_ratio": fills / orders if orders else np.nan,
        "terminal_inv": abs(sim.q),
        "liquidation_cost": liquidation_cost,
        "penalized_return": total_reward,
    }


def evaluate_policy(policy, params: SimulatorParams, seeds) -> pd.DataFrame:
    return pd.DataFrame([run_episode(policy, params, s) for s in seeds])


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard error of each metric, one row per strategy."""
    rows = {}
    for name, df in results.items():
        row = {}
        for col in df.columns:
            row[col] = df[col].mean()
            row[f"{col}_se"] = df[col].std(ddof=1) / np.sqrt(len(df))
        # Sharpe-like ratio across episodes: mean PnL / std PnL.
        row["pnl_sharpe"] = df["pnl"].mean() / df["pnl"].std(ddof=1)
        rows[name] = row
    return pd.DataFrame(rows).T
