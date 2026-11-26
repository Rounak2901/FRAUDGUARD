"""
Bayesian entity-risk beliefs as Beta(α, β) with optional daily decay.

Prior mean ≈ α0 / (α0 + β0) ≈ global fraud rate.
Fraud evidence: α += 1; genuine: β += 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from scipy.stats import beta as beta_dist


@dataclass
class EntityBelief:
    entity_id: str
    alpha: float
    beta: float
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        lo = (1 - level) / 2
        hi = 1 - lo
        return float(beta_dist.ppf(lo, self.alpha, self.beta)), float(beta_dist.ppf(hi, self.alpha, self.beta))


class EntityRiskStore:
    """In-memory Beta beliefs keyed by entity_id (card / user / merchant)."""

    def __init__(self, alpha0: float = 1.0, beta0: float = 500.0, decay_per_day: float = 0.99) -> None:
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.decay_per_day = decay_per_day
        self._beliefs: Dict[str, EntityBelief] = {}
        self._history: Dict[str, list] = {}

    def get(self, entity_id: str) -> EntityBelief:
        if entity_id not in self._beliefs:
            self._beliefs[entity_id] = EntityBelief(entity_id, self.alpha0, self.beta0)
        return self._beliefs[entity_id]

    def _apply_decay(self, belief: EntityBelief, now: datetime) -> None:
        if belief.updated_at.tzinfo is None:
            belief.updated_at = belief.updated_at.replace(tzinfo=timezone.utc)
        days = max(0.0, (now - belief.updated_at).total_seconds() / 86400.0)
        if days <= 0:
            return
        factor = self.decay_per_day ** days
        # Shrink toward prior
        belief.alpha = self.alpha0 + (belief.alpha - self.alpha0) * factor
        belief.beta = self.beta0 + (belief.beta - self.beta0) * factor

    def peek(self, entity_id: str) -> dict:
        b = self.get(entity_id)
        lo, hi = b.credible_interval()
        return {
            "entity_id": entity_id,
            "alpha": b.alpha,
            "beta": b.beta,
            "mean": b.mean,
            "variance": b.variance,
            "credible_interval_95": [lo, hi],
            "updated_at": b.updated_at.isoformat(),
        }

    def update(self, entity_id: str, is_fraud: bool, timestamp: Optional[datetime] = None) -> EntityBelief:
        now = timestamp or datetime.now(timezone.utc)
        b = self.get(entity_id)
        self._apply_decay(b, now)
        if is_fraud:
            b.alpha += 1.0
        else:
            b.beta += 1.0
        b.updated_at = now
        self._history.setdefault(entity_id, []).append(
            {"timestamp": now.isoformat(), "alpha": b.alpha, "beta": b.beta, "mean": b.mean, "is_fraud": is_fraud}
        )
        return b

    def history(self, entity_id: str) -> list:
        return list(self._history.get(entity_id, []))
