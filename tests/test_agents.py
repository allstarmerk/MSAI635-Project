"""Tests for the tabular agents and evaluation metrics."""
import numpy as np
import pytest

from market_making.agents import TabularAgent
from market_making.baselines import FixedSpreadBaseline
from market_making.evaluation import run_episode
from market_making.simulator import MarketSimulator, SimulatorParams


def test_state_index_covers_full_range():
    agent = TabularAgent("sarsa", q_max=20, T=200, n_time_bins=10)
    assert agent.state_index(-20, 0) == (0, 0)
    assert agent.state_index(20, 199) == (40, 9)
    assert agent.state_index(0, 100) == (20, 5)


def test_q_learning_and_sarsa_differ_only_in_bootstrap_target():
    s, s_next, a, a_next = (20, 0), (21, 1), 0, 4
    agents = {m: TabularAgent(m, q_max=20, T=200) for m in ("q_learning", "sarsa")}
    for ag in agents.values():
        ag.Q[s_next] = np.arange(9, dtype=float)  # max = 8, Q[s_next][a_next] = 4
        ag.update(s, a, 1.0, s_next, a_next, done=False)  # first visit -> lr = 1
    assert agents["q_learning"].Q[s + (a,)] == pytest.approx(1.0 + 8.0)
    assert agents["sarsa"].Q[s + (a,)] == pytest.approx(1.0 + 4.0)


def test_terminal_update_does_not_bootstrap():
    agent = TabularAgent("q_learning", q_max=20, T=200)
    agent.Q[:] = 100.0
    agent.visits[:] = 0
    agent.update((20, 9), 3, -0.5, None, None, done=True)
    assert agent.Q[20, 9, 3] == pytest.approx(-0.5)


def test_run_episode_pnl_matches_liquidated_wealth():
    params = SimulatorParams()
    m = run_episode(FixedSpreadBaseline(2), params, seed=3)
    sim = MarketSimulator(params, seed=3)
    sim.reset()
    for _ in range(params.T):
        sim.step(2, 2)
    expected = sim.X + sim.q * sim.S - params.c * abs(sim.q)
    assert m["pnl"] == pytest.approx(expected)
    assert m["terminal_inv"] == abs(sim.q)
    assert 0.0 <= m["fill_ratio"] <= 1.0


def test_agent_learns_to_reduce_inventory_risk():
    params = SimulatorParams()
    agent = TabularAgent("sarsa", params.q_max, params.T, seed=0)
    agent.train(MarketSimulator(params, seed=1), 2000)
    seeds = range(500_000, 500_100)
    agent_rms = np.mean([run_episode(agent, params, s)["rms_inventory"] for s in seeds])
    base_rms = np.mean([run_episode(FixedSpreadBaseline(2), params, s)["rms_inventory"] for s in seeds])
    assert agent_rms < base_rms
