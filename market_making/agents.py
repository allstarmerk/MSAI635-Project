from __future__ import annotations

import numpy as np

from market_making.env import ACTIONS
from market_making.simulator import MarketSimulator

METHODS = ("q_learning", "sarsa")


class TabularAgent:
    def __init__(
        self,
        method: str,
        q_max: int,
        T: int,
        n_time_bins: int = 10,
        gamma: float = 1.0,
        lr_min: float = 0.01,
        lr_power: float = 0.6,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_frac: float = 0.5,
        seed: int | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.q_max = q_max
        self.T = T
        self.n_time_bins = n_time_bins
        self.gamma = gamma
        self.lr_min = lr_min
        self.lr_power = lr_power
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_frac = eps_decay_frac
        self.rng = np.random.default_rng(seed)

        shape = (2 * q_max + 1, n_time_bins, len(ACTIONS))
        self.Q = np.zeros(shape)
        self.visits = np.zeros(shape, dtype=np.int64)

    #state / action helpers 
    def state_index(self, q: int, t: int) -> tuple[int, int]:
        time_bin = min(t * self.n_time_bins // self.T, self.n_time_bins - 1)
        return q + self.q_max, time_bin

    def greedy_action(self, s: tuple[int, int]) -> int:
        q_vals = self.Q[s]
        best = np.flatnonzero(q_vals == q_vals.max())
        return int(self.rng.choice(best)) if len(best) > 1 else int(best[0])

    def select_action(self, s: tuple[int, int], eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(len(ACTIONS)))
        return self.greedy_action(s)
