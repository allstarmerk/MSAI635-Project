from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_making.agents import METHODS, TabularAgent
from market_making.baselines import FixedSpreadBaseline
from market_making.env import ACTIONS
from market_making.evaluation import evaluate_policy, summarize
from market_making.simulator import MarketSimulator, SimulatorParams

EVAL_SEED_OFFSET = 1_000_000  # held-out seeds never used in training


def plot_policy(agent: TabularAgent, path: Path, title: str) -> None:
    greedy = agent.Q.argmax(axis=2)
    visited = agent.visits.sum(axis=2) > 0
    d_bid = np.where(visited, np.array([a[0] for a in ACTIONS])[greedy], np.nan)
