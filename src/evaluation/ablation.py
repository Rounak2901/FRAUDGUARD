"""Ablation study helpers — contribution of each system component."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import binary_metrics


def run_ablation(
    y_true: np.ndarray,
    components: dict[str, np.ndarray],
    amounts: np.ndarray,
    c_fp: float = 5.0,
) -> dict[str, Any]:
    """
    components keys expected:
      nn, nn_unc_gated (optional), entity, anomaly, ensemble
    """
    out = {}
    for name, scores in components.items():
        m = binary_metrics(y_true, scores)
        # Simple cost at 0.5
        pred = (scores >= 0.5).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn_mask = (pred == 0) & (y_true == 1)
        cost = float(fp * c_fp + amounts[fn_mask].sum())
        m["business_cost_at_0_5"] = cost
        out[name] = m
    return out
