"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    transaction_id: Optional[str] = None
    amount: float = Field(..., ge=0)
    time: float = Field(0, description="Seconds from dataset origin (ULB Time)")
    entity_id: str = "card_demo"
    merchant_id: str = "merch_demo"
    # Optional V1..V28 — if omitted, zeros are used (demo form)
    features: Optional[dict[str, float]] = None
    true_label: Optional[int] = None


class ReasonOut(BaseModel):
    feature: str
    impact: float
    text: str


class ScoreOut(BaseModel):
    transaction_id: str
    p_fraud: float
    uncertainty_std: float
    credible_interval_95: list[float]
    anomaly_score: float
    entity_risk_posterior: float
    risk_score: float
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    expected_loss_if_approved: float
    top_reasons: list[ReasonOut]
    latency_ms: float
    model_version: str
    amount: float | None = None
    entity_id: str | None = None
    merchant_id: str | None = None


class BatchScoreIn(BaseModel):
    transactions: list[TransactionIn]


class FeedbackIn(BaseModel):
    transaction_id: str
    analyst_label: Literal[0, 1]  # 0 genuine, 1 fraud


class StreamControl(BaseModel):
    rate_tps: float = Field(5, ge=1, le=100)
    dataset_slice: Literal["train", "val", "test"] = "test"


class CostSimIn(BaseModel):
    c_fp: float = 5.0
    c_fn_multiplier: float = 1.0
    c_review: float = 2.0


class HealthOut(BaseModel):
    status: str
    model_version: str
    bundle_loaded: bool
