# Project Status — Stage 1 (Environment)

## Context

Kickoff nudge from Yating Zhang (GitHub: `yating-zhang-01`) on 2026-10-25: "Let's get this project moving. The goal is to create an environment," quoting Section 4 of the interim draft (`Draft (1) (1).pdf`), which claims the simulator was already verified for cash/inventory accounting, Poisson arrival rates, fill-probability decay, and Brownian price increments.

Stage 1 scope = build the synthetic market environment and back up those Section 4 claims with actual code and tests — not the full RL training / baseline comparison, which is next-phase work.

## What's built

- `market_making/simulator.py` — core synthetic market: Brownian mid-price, Poisson order arrivals, `exp(-k*d)` fill probability, inventory/cash accounting, hard inventory cap, quadratic inventory penalty, terminal liquidation cost. Parameters match the draft's table (S0=100, tick=0.01, λ=1/step, k=0.5, T=200, q_max=20, actions ∈ {1,2,3} ticks).
- `market_making/env.py` — Gymnasium wrapper, 9 discrete actions (d_bid, d_ask ∈ {1,2,3}).
- `market_making/baselines.py` — `FixedSpreadBaseline`, the one baseline required for stage 1. (The optional inventory-aware heuristic baseline was drafted and then pulled back out to keep this stage's scope tight — it's "if time permits" per the report, not required yet.)
- `tests/test_simulator.py` — 4 tests reproducing the exact verification claims from Section 4; all passing.
- `requirements.txt` + setup instructions in `README.md`.

## Verified (via `pytest tests/`)

- Cash/inventory accounting is exact.
- Empirical Poisson arrival rates match the specified λ.
- Fill probability decreases with quote distance, matching `exp(-k*d)`.
- Brownian price increments have ~zero mean and the expected variance.

## Intentionally not yet built (next phase)

- Q-learning / DQN training loop.
- Evaluation metrics reporting (PnL, RMS inventory, fill rate, Sharpe-like ratio).
- Any plots/analysis.
- Optional inventory-aware heuristic baseline / Avellaneda-Stoikov benchmark.

## Open item

Haven't confirmed PyTorch / Stable-Baselines3 install compatibility with our local Python 3.14 (Section 2.5 libraries) — worth checking before starting agent training, since newer Python releases sometimes lag behind PyTorch/SB3 wheel support.

## Repo state

`market_making/`, `tests/`, `requirements.txt`, and this file are local and untracked — nothing from stage 1 has been committed or pushed yet.
