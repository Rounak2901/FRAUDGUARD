"""
Time-based train / validation / test splits.

Why not random splits?
  Transaction fraud has temporal structure (concept drift, attacker campaigns).
  A random split lets the model train on future fraud patterns and then 'predict'
  them on earlier rows — information leakage that inflates PR-AUC unrealistically.
  Sorting by Time and taking earliest 70% / next 15% / final 15% respects causality.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "Time",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by ``time_col`` ascending and cut into train / val / test contiguous blocks.

    Ratios must sum to 1.0 (within floating tolerance).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    n = len(sorted_df)
    i_train = int(n * train_ratio)
    i_val = i_train + int(n * val_ratio)

    train = sorted_df.iloc[:i_train].copy()
    val = sorted_df.iloc[i_train:i_val].copy()
    test = sorted_df.iloc[i_val:].copy()
    return train, val, test


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target: str = "Class") -> dict:
    """Return counts and fraud rates for documentation / logging."""

    def _stats(part: pd.DataFrame, name: str) -> dict:
        n = len(part)
        fraud = int(part[target].sum()) if target in part.columns else 0
        return {
            "name": name,
            "n": n,
            "fraud": fraud,
            "fraud_rate": fraud / n if n else 0.0,
            "time_min": float(part["Time"].min()) if "Time" in part.columns and n else None,
            "time_max": float(part["Time"].max()) if "Time" in part.columns and n else None,
        }

    return {
        "train": _stats(train, "train"),
        "val": _stats(val, "val"),
        "test": _stats(test, "test"),
    }
