"""
Causal (leakage-free) entity rolling features for streaming / PaySim.

All window aggregates use only past transactions for that entity
(strictly timestamp < current). Suitable for both offline rebuild and
online FeatureStore updates in the API.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class _EntityState:
    """In-memory history for one entity (card / user)."""

    times: Deque[float] = field(default_factory=deque)
    amounts: Deque[float] = field(default_factory=deque)
    merchants: Deque[str] = field(default_factory=deque)

    def push(self, t: float, amount: float, merchant: str) -> None:
        self.times.append(t)
        self.amounts.append(amount)
        self.merchants.append(merchant)

    def decay_before(self, t_now: float, window_seconds: float) -> None:
        while self.times and (t_now - self.times[0]) > window_seconds:
            self.times.popleft()
            self.amounts.popleft()
            self.merchants.popleft()


class OnlineFeatureBuilder:
    """
    Maintains per-entity rolling stats for real-time scoring.

    Features produced (when history exists):
      tx_count_1h, tx_count_24h, amount_mean, amount_std,
      time_since_last, amount_zscore, first_time_merchant, velocity_1h
    """

    def __init__(self, window_hours: List[float] | None = None) -> None:
        self.window_hours = window_hours or [1.0, 24.0]
        self._states: Dict[str, _EntityState] = defaultdict(_EntityState)
        self._seen_merchants: Dict[str, set] = defaultdict(set)

    def transform_one(
        self,
        entity_id: str,
        timestamp: float,
        amount: float,
        merchant_id: str = "unknown",
        update: bool = True,
    ) -> dict:
        """Compute features from past only, then optionally append this tx."""
        st = self._states[entity_id]
        # Snapshots before update
        times = list(st.times)
        amounts = list(st.amounts)

        feat = {
            "tx_count_1h": 0,
            "tx_count_24h": 0,
            "amount_mean": 0.0,
            "amount_std": 0.0,
            "time_since_last": -1.0,
            "amount_zscore": 0.0,
            "first_time_merchant": 1.0,
            "velocity_1h": 0.0,
        }

        if times:
            arr_t = np.asarray(times, dtype=float)
            arr_a = np.asarray(amounts, dtype=float)
            feat["tx_count_1h"] = int(np.sum((timestamp - arr_t) <= 3600))
            feat["tx_count_24h"] = int(np.sum((timestamp - arr_t) <= 86400))
            feat["amount_mean"] = float(arr_a.mean())
            feat["amount_std"] = float(arr_a.std(ddof=0)) if len(arr_a) > 1 else 0.0
            feat["time_since_last"] = float(timestamp - times[-1])
            if feat["amount_std"] > 1e-9:
                feat["amount_zscore"] = float((amount - feat["amount_mean"]) / feat["amount_std"])
            else:
                feat["amount_zscore"] = 0.0
            feat["velocity_1h"] = float(feat["tx_count_1h"])  # txs per hour window
            feat["first_time_merchant"] = 0.0 if merchant_id in self._seen_merchants[entity_id] else 1.0

        if update:
            st.push(timestamp, amount, merchant_id)
            self._seen_merchants[entity_id].add(merchant_id)
            # Bound memory: keep ~24h worth assuming dense streams; hard cap 500
            while len(st.times) > 500:
                st.times.popleft()
                st.amounts.popleft()
                st.merchants.popleft()

        return feat


def build_entity_features_offline(
    df: pd.DataFrame,
    entity_col: str = "entity_id",
    time_col: str = "Time",
    amount_col: str = "Amount",
    merchant_col: str = "merchant_id",
) -> pd.DataFrame:
    """
    Rebuild causal entity features for an entire chronologically sorted frame.
    """
    builder = OnlineFeatureBuilder()
    rows = []
    ordered = df.sort_values(time_col)
    for _, r in ordered.iterrows():
        merchant = str(r[merchant_col]) if merchant_col in df.columns else "unknown"
        feat = builder.transform_one(
            entity_id=str(r[entity_col]),
            timestamp=float(r[time_col]),
            amount=float(r[amount_col]),
            merchant_id=merchant,
            update=True,
        )
        rows.append(feat)
    feat_df = pd.DataFrame(rows, index=ordered.index)
    return df.join(feat_df)
