# MSAI635-Project

## Overview

**Reinforcement Learning for Inventory-Aware Market Making in a Synthetic Market**

This project trains an RL agent to set bid/ask quotes in a synthetic market, aiming for a better profitability-vs-inventory-risk tradeoff than a fixed-spread baseline market maker. The environment is a simulated limit order book with Brownian mid-price dynamics, Poisson order arrivals, and distance-dependent fill probabilities; the agent (tabular Q-learning / DQN, with PPO as a possible extension) observes inventory and time-remaining and chooses discrete bid/ask tick distances to maximize PnL net of an inventory-risk penalty.

Team: Johnathan Sheikh, Chenhongshu Yu, Yating Zhang (Group 14).

## Status

The synthetic market simulator (`market_making/`) is implemented and verified: cash/inventory accounting, Poisson arrival rates, distance-dependent fill probability, and Brownian price increments all check out against the report's Section 4 claims (`tests/test_simulator.py`, all passing). A Gymnasium environment wrapper and the fixed-spread baseline are in place; training the RL agent and comparing against the baseline is next.

## Setup

```
python -m venv .venv
.venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```