"""Synthetic market simulator for inventory-aware market making.

Implements the model from the project's interim report (Section 2.1-2.3):
Brownian mid-price, Poisson order arrivals, distance-dependent fill
probability, and inventory/cash accounting with quadratic inventory
penalty and terminal liquidation cost.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulatorParams:
    S0: float = 100.0
    tick_size: float = 0.01
    sigma_ticks: float = 1.0       # price volatility, in ticks/step
    lambda_buy: float = 1.0        # buy order arrival rate, per step
    lambda_sell: float = 1.0       # sell order arrival rate, per step
    k: float = 0.5                 # fill-probability decay
    T: int = 200                   # episode horizon (steps)
    q_max: int = 20                # hard inventory limit
    c_ticks: float = 1.0           # terminal liquidation cost, in ticks/unit
    beta: float = 0.1              # inventory-penalty coefficient

    @property
    def sigma(self) -> float:
        return self.sigma_ticks * self.tick_size

    @property
    def c(self) -> float:
        return self.c_ticks * self.tick_size


class MarketSimulator:
    """Core synthetic market used to train and evaluate market-making strategies."""

    def __init__(self, params: SimulatorParams | None = None, seed: int | None = None):
        self.params = params or SimulatorParams()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def seed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self):
        p = self.params
        self.t = 0
        self.S = p.S0
        self.q = 0
        self.X = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        p = self.params
        return np.array(
            [self.q / p.q_max, (p.T - self.t) / p.T],
            dtype=np.float64,
        )

    @property
    def wealth(self) -> float:
        return self.X + self.q * self.S

    def step(self, d_bid: int, d_ask: int):
        """Advance the simulator by one step given quote distances (in ticks)."""
        p = self.params
        if self.t >= p.T:
            raise RuntimeError("step() called after episode has terminated; call reset()")

        S_t = self.S
        q_t = self.q
        X_t = self.X
        W_t = X_t + q_t * S_t

        bid_price = S_t - p.tick_size * d_bid
        ask_price = S_t + p.tick_size * d_ask

        n_buy = self.rng.poisson(p.lambda_buy)
        n_sell = self.rng.poisson(p.lambda_sell)

        p_ask_fill = np.exp(-p.k * d_ask)
        p_bid_fill = np.exp(-p.k * d_bid)

        f_ask = self.rng.binomial(n_buy, p_ask_fill) if n_buy > 0 else 0
        f_bid = self.rng.binomial(n_sell, p_bid_fill) if n_sell > 0 else 0

        # Hard inventory limit: clip fills that would breach +/- q_max.
        f_bid = min(f_bid, max(p.q_max - q_t, 0))
        f_ask = min(f_ask, max(q_t + p.q_max, 0))

        q_next = q_t + f_bid - f_ask
        X_next = X_t - f_bid * bid_price + f_ask * ask_price

        eps = self.rng.normal()
        S_next = S_t + p.sigma * eps

        self.S, self.q, self.X = S_next, q_next, X_next
        self.t += 1

        W_next = X_next + q_next * S_next
        inv_penalty = p.beta * (q_next / p.q_max) ** 2
        reward = (W_next - W_t) - inv_penalty

        done = self.t >= p.T
        liquidation_cost = 0.0
        if done:
            liquidation_cost = p.c * abs(q_next)
            reward -= liquidation_cost

        info = {
            "n_buy": n_buy,
            "n_sell": n_sell,
            "f_bid": f_bid,
            "f_ask": f_ask,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "wealth": W_next,
            "liquidation_cost": liquidation_cost,
        }
        return self._state(), reward, done, info
