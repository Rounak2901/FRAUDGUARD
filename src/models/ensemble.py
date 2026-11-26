"""Stack / grid-search ensemble weights for NN + entity + anomaly."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def grid_search_weights(
    p_nn: np.ndarray,
    entity: np.ndarray,
    anomaly: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> dict:
    best = {"w_nn": 0.55, "w_entity": 0.25, "w_anomaly": 0.20, "pr_auc": -1.0}
    for w1, w2, w3 in itertools.product(grid, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) > 1e-9:
            continue
        score = w1 * p_nn + w2 * entity + w3 * anomaly
        pr = float(average_precision_score(y, score)) if y.sum() else 0.0
        if pr > best["pr_auc"]:
            best = {"w_nn": w1, "w_entity": w2, "w_anomaly": w3, "pr_auc": pr}
    return best


def combine(p_nn: np.ndarray, entity: np.ndarray, anomaly: np.ndarray, weights: dict) -> np.ndarray:
    return weights["w_nn"] * p_nn + weights["w_entity"] * entity + weights["w_anomaly"] * anomaly
