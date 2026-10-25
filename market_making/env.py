"""Gymnasium wrapper around MarketSimulator (Section 2.5 of the interim report)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from market_making.simulator import MarketSimulator, SimulatorParams

QUOTE_DISTANCES = (1, 2, 3)
ACTIONS = [(d_bid, d_ask) for d_bid in QUOTE_DISTANCES for d_ask in QUOTE_DISTANCES]


class MarketMakingEnv(gym.Env):
    """Discrete 9-action (d_bid, d_ask in {1,2,3} ticks) market-making environment."""

    metadata = {"render_modes": []}

    def __init__(self, params: SimulatorParams | None = None, seed: int | None = None):
        super().__init__()
        self.sim = MarketSimulator(params=params, seed=seed)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        state = self.sim.reset()
        return state.astype(np.float32), {}

    def step(self, action: int):
        d_bid, d_ask = ACTIONS[action]
        state, reward, done, info = self.sim.step(d_bid, d_ask)
        return state.astype(np.float32), reward, done, False, info
