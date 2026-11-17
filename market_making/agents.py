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

    def act(self, q: int, q_max: int, t: int, T: int) -> tuple[int, int]:
        """Greedy policy, same interface as the baselines."""
        return ACTIONS[self.greedy_action(self.state_index(q, t))]

    def epsilon(self, episode: int, n_episodes: int) -> float:
        decay_eps = max(1, int(self.eps_decay_frac * n_episodes))
        frac = min(episode / decay_eps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    #learning 
    def update(self, s, a, r, s_next, a_next, done: bool) -> None:
        if done:
            target = r
        elif self.method == "q_learning":
            target = r + self.gamma * self.Q[s_next].max()
        else:  # sarsa
            target = r + self.gamma * self.Q[s_next][a_next]

        sa = s + (a,)
        self.visits[sa] += 1
        # Visit count learning rate: rewards are noisy (inventory * price move),
        # so a decaying step size averages the noise out; floored at lr_min.
        lr = max(self.lr_min, self.visits[sa] ** -self.lr_power)
        self.Q[sa] += lr * (target - self.Q[sa])

    def train(self, sim: MarketSimulator, n_episodes: int) -> np.ndarray:
        """Train on the simulator; returns the (penalized) return of each episode."""
        returns = np.empty(n_episodes)
        for ep in range(n_episodes):
            eps = self.epsilon(ep, n_episodes)
            sim.reset()
            s = self.state_index(sim.q, sim.t)
            a = self.select_action(s, eps)
            total, done = 0.0, False
            while not done:
                d_bid, d_ask = ACTIONS[a]
                _, r, done, _ = sim.step(d_bid, d_ask)
                total += r
                if done:
                    self.update(s, a, r, None, None, True)
                    break
                s_next = self.state_index(sim.q, sim.t)
                a_next = self.select_action(s_next, eps)
                self.update(s, a, r, s_next, a_next, False)
                s, a = s_next, a_next
            returns[ep] = total
        return returns

    def save(self, path) -> None:
        np.savez(path, Q=self.Q, visits=self.visits, method=self.method,
                 q_max=self.q_max, T=self.T, n_time_bins=self.n_time_bins)

    @classmethod
    def load(cls, path) -> "TabularAgent":
        data = np.load(path)
        agent = cls(str(data["method"]), int(data["q_max"]), int(data["T"]),
                    n_time_bins=int(data["n_time_bins"]))
        agent.Q, agent.visits = data["Q"], data["visits"]
        return agent
