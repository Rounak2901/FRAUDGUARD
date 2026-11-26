"""Latency benchmark helpers for the scoring endpoint."""

from __future__ import annotations

import statistics
import time
from typing import Callable


def benchmark_callable(fn: Callable[[], None], n: int = 100, warmup: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "n": n,
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95) - 1],
        "p99_ms": times[int(len(times) * 0.99) - 1],
        "mean_ms": float(statistics.mean(times)),
        "tps": 1000.0 / max(statistics.mean(times), 1e-9),
    }
