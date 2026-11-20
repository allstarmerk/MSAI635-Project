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
    d_ask = np.where(visited, np.array([a[1] for a in ACTIONS])[greedy], np.nan)  ####
    panels = [("d_bid (ticks)", d_bid, "viridis"), ("d_ask (ticks)", d_ask, "viridis"),
              ("skew = d_bid - d_ask", d_bid - d_ask, "coolwarm")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    extent = [0, agent.T, -agent.q_max - 0.5, agent.q_max + 0.5]
    for ax, (name, data, cmap) in zip(axes, panels):
        lim = (-2, 2) if "skew" in name else (1, 3)
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent,
                       cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_title(name)
        ax.set_xlabel("time step t")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("inventory q")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=30_000, help="training episodes per agent")
    ap.add_argument("--eval-episodes", type=int, default=1_000, help="held-out evaluation episodes")
    ap.add_argument("--zeta", type=float, nargs="+", default=[0.1], help="inventory-penalty coefficient(s)")
    ap.add_argument("--time-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    