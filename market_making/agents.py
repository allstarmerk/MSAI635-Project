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
    