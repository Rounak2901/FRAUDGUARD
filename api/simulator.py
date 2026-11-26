"""Replay test-set transactions at a configurable TPS into the scoring pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

import pandas as pd

from api.schemas import TransactionIn
from api.services.scoring import scoring_service
from src.config import load_config, resolve_path


class StreamSimulator:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self.running = False
        self.rate_tps = 5.0
        self.dataset_slice = "test"
        self._df: Optional[pd.DataFrame] = None
        self._idx = 0
        self.subscribers: list[Callable[[dict], Awaitable[None]]] = []

    def load_slice(self, slice_name: str = "test") -> None:
        cfg = load_config()
        path = resolve_path(cfg, "processed_dir") / f"{slice_name}.parquet"
        self._df = pd.read_parquet(path)
        self.dataset_slice = slice_name
        self._idx = 0
        print(f"[sim] Loaded {len(self._df)} rows from {slice_name}")

    def subscribe(self, cb: Callable[[dict], Awaitable[None]]) -> None:
        self.subscribers.append(cb)

    def unsubscribe(self, cb: Callable[[dict], Awaitable[None]]) -> None:
        if cb in self.subscribers:
            self.subscribers.remove(cb)

    async def start(self, rate_tps: float = 5.0, dataset_slice: str = "test") -> dict:
        if self.running:
            self.rate_tps = rate_tps
            return {"status": "already_running", "rate_tps": self.rate_tps}
        if self._df is None or self.dataset_slice != dataset_slice:
            self.load_slice(dataset_slice)
        self.rate_tps = rate_tps
        self.running = True
        self._task = asyncio.create_task(self._loop())
        return {"status": "started", "rate_tps": self.rate_tps, "slice": self.dataset_slice}

    async def stop(self) -> dict:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return {"status": "stopped"}

    async def _loop(self) -> None:
        assert self._df is not None
        feature_cols = [c for c in self._df.columns if c.startswith("V")]
        while self.running:
            if self._idx >= len(self._df):
                self._idx = 0
            row = self._df.iloc[self._idx]
            self._idx += 1
            feats = {c: float(row[c]) for c in feature_cols}
            tx = TransactionIn(
                transaction_id=f"sim_{self._idx:06d}",
                amount=float(row["Amount"]),
                time=float(row["Time"]),
                entity_id=str(row.get("entity_id", row.get("card_id", "card_demo"))),
                merchant_id=str(row.get("merchant_id", "merch_demo")),
                features=feats,
                true_label=int(row["Class"]) if "Class" in row else None,
            )
            result = scoring_service.score(tx)
            payload = result.model_dump()
            payload["true_label"] = tx.true_label
            for cb in list(self.subscribers):
                try:
                    await cb(payload)
                except Exception:
                    pass
            await asyncio.sleep(max(0.01, 1.0 / max(self.rate_tps, 0.1)))


simulator = StreamSimulator()
