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
