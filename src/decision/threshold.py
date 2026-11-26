"""Optimise Approve/Block threshold on validation to minimise total business cost."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def total_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    c_fp: float,
    c_fn_multiplier: float = 1.0,
) -> float:
    pred = (y_prob >= threshold).astype(int)
    fp = (pred == 1) & (y_true == 0)
    fn = (pred == 0) & (y_true == 1)
    return float(fp.sum() * c_fp + (amounts[fn] * c_fn_multiplier).sum())


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    c_fp: float = 5.0,
    c_fn_multiplier: float = 1.0,
    grid: Iterable[float] | None = None,
) -> dict:
    grid = list(grid) if grid is not None else list(np.linspace(0.05, 0.95, 37))
    best_t, best_c = 0.5, float("inf")
    curve = []
    for t in grid:
        c = total_cost(y_true, y_prob, amounts, t, c_fp, c_fn_multiplier)
        curve.append({"threshold": float(t), "cost": c})
        if c < best_c:
            best_c, best_t = c, float(t)
    default_cost = total_cost(y_true, y_prob, amounts, 0.5, c_fp, c_fn_multiplier)
    return {
        "optimal_threshold": best_t,
        "optimal_cost": best_c,
        "default_0_5_cost": default_cost,
        "cost_saved_vs_0_5": default_cost - best_c,
        "curve": curve,
    }
