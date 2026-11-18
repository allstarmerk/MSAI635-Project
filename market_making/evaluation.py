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
