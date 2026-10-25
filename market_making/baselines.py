"""Baseline quoting strategies (Section 3 of the interim report)."""
from __future__ import annotations


class FixedSpreadBaseline:
    """Always quotes the same bid/ask distance, regardless of inventory or time."""

    def __init__(self, distance: int = 2):
        self.distance = distance

    def act(self, q: int, q_max: int, t: int, T: int) -> tuple[int, int]:
        return self.distance, self.distance
