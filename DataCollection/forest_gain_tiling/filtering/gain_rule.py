"""Programmable forest-gain pixel rule, shared by generate_aois.py (AOI gate)
and filtering/raster_stats.py (tile filtering).

A gain pixel always starts non-forest: cover_start < settings.non_tree_threshold_frac.
What it must reach by the end year depends on the rule:

  threshold  (default, the original rule)
             cover_end > settings.min_tree_threshold_frac

  delta      cover_end - cover_start >= GAIN_MIN_DELTA   (percentage points)

An AOI / tile passes if gain pixels make up at least the configured fraction
of its area (generate_aois.MIN_GAIN_FRACTION, settings.gain_pct_min). Nothing is
averaged.

Configuration (first match wins: environment variable, then settings, then default):

  GAIN_RULE       / settings.gain_rule        "threshold" | "delta"   (threshold)
  GAIN_MIN_DELTA  / settings.gain_min_delta   points                  (50)

    GAIN_RULE=delta GAIN_MIN_DELTA=50 python generate_aois.py
"""

from __future__ import annotations

import os

import ee
from config import settings

RULES = ("threshold", "delta")


def _opt(env: str, attr: str, default):
    raw = os.getenv(env)
    if raw not in (None, ""):
        return raw
    return getattr(settings, attr, default)


def gain_rule() -> str:
    rule = str(_opt("GAIN_RULE", "gain_rule", "threshold")).strip().lower()
    if rule not in RULES:
        raise ValueError(f"Unknown gain rule '{rule}'. Use one of {RULES}.")
    return rule


def min_delta() -> float:
    return float(_opt("GAIN_MIN_DELTA", "gain_min_delta", 50.0))


def describe() -> str:
    if gain_rule() == "threshold":
        return (
            f"gain rule: threshold (start < {settings.non_tree_threshold_frac:g}, "
            f"end > {settings.min_tree_threshold_frac:g})"
        )
    return (
        f"gain rule: delta (start < {settings.non_tree_threshold_frac:g}, "
        f"rise >= {min_delta():g})"
    )


def gain_image(cover_start: ee.Image, cover_end: ee.Image) -> ee.Image:
    """0/1 band 'gain'. Inputs are tree-cover percent (0-100). Reduce with
    ee.Reducer.mean() to get the gain-pixel fraction."""
    non_forest_start = cover_start.lt(settings.non_tree_threshold_frac)

    if gain_rule() == "threshold":
        reached = cover_end.gt(settings.min_tree_threshold_frac)
    else:
        reached = cover_end.subtract(cover_start).gte(min_delta())

    return non_forest_start.And(reached).unmask(0).rename("gain")
