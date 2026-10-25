"""Verification tests matching the interim report, Section 4:

'We tested cash and inventory accounting, confirmed that empirical Poisson
arrival rates are close to their specified values, verified that fill
probability decreases with quote distance, and confirmed that Brownian
price increments have approximately zero mean and sufficient variance.'
"""
import numpy as np
import pytest

from market_making.simulator import MarketSimulator, SimulatorParams


def test_cash_and_inventory_accounting():
    sim = MarketSimulator(params=SimulatorParams(), seed=0)
    sim.reset()
    for _ in range(50):
        S_t, q_t, X_t = sim.S, sim.q, sim.X
        d_bid, d_ask = 2, 2
        bid_price = S_t - sim.params.tick_size * d_bid
        ask_price = S_t + sim.params.tick_size * d_ask

        state, reward, done, info = sim.step(d_bid, d_ask)

        f_bid, f_ask = info["f_bid"], info["f_ask"]
        expected_q = q_t + f_bid - f_ask
        expected_X = X_t - f_bid * bid_price + f_ask * ask_price

        assert sim.q == expected_q
        assert sim.X == pytest.approx(expected_X)
        assert abs(sim.q) <= sim.params.q_max
        if done:
            break


def test_poisson_arrival_rates_match_specified_value():
    lam_buy, lam_sell = 1.3, 0.7
    params = SimulatorParams(lambda_buy=lam_buy, lambda_sell=lam_sell, T=20000)
    sim = MarketSimulator(params=params, seed=1)
    sim.reset()

    n_buys, n_sells = [], []
    for _ in range(params.T):
        _, _, done, info = sim.step(2, 2)
        n_buys.append(info["n_buy"])
        n_sells.append(info["n_sell"])
        if done:
            break

    assert np.mean(n_buys) == pytest.approx(lam_buy, abs=0.05)
    assert np.mean(n_sells) == pytest.approx(lam_sell, abs=0.05)


def test_fill_probability_decreases_with_quote_distance():
    empirical_fill_rate = {}
    for d in (1, 2, 3):
        params = SimulatorParams(lambda_buy=5.0, lambda_sell=5.0, k=0.5, T=20000)
        sim = MarketSimulator(params=params, seed=42)
        sim.reset()

        total_sell_orders, total_bid_fills = 0, 0
        for _ in range(params.T):
            _, _, done, info = sim.step(d, d)
            total_sell_orders += info["n_sell"]
            total_bid_fills += info["f_bid"]
            if done:
                break

        empirical_fill_rate[d] = total_bid_fills / total_sell_orders

    assert empirical_fill_rate[1] > empirical_fill_rate[2] > empirical_fill_rate[3]
    for d, rate in empirical_fill_rate.items():
        assert rate == pytest.approx(np.exp(-0.5 * d), abs=0.03)


def test_brownian_price_increments_zero_mean_and_expected_variance():
    sigma_ticks = 1.5
    params = SimulatorParams(sigma_ticks=sigma_ticks, T=20000)
    sim = MarketSimulator(params=params, seed=7)
    sim.reset()

    prices = [sim.S]
    for _ in range(params.T):
        _, _, done, _ = sim.step(2, 2)
        prices.append(sim.S)
        if done:
            break

    increments = np.diff(prices)
    expected_sigma = sigma_ticks * params.tick_size

    assert np.mean(increments) == pytest.approx(0.0, abs=3 * expected_sigma / np.sqrt(len(increments)))
    assert np.std(increments) == pytest.approx(expected_sigma, rel=0.05)
