"""
Cost-sensitive Approve / Review / Block engine.

Block when expected loss of approving (p * amount * c_fn_mult) exceeds
expected friction of blocking ((1-p) * c_fp). Review when uncertainty is high
or probability sits near the decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["APPROVE", "REVIEW", "BLOCK"]


@dataclass
class DecisionConfig:
    c_fp: float = 5.0
    c_fn_multiplier: float = 1.0
    c_review: float = 2.0
    review_uncertainty_threshold: float = 0.12
    review_margin: float = 0.05
    block_threshold: float = 0.5


@dataclass
class DecisionResult:
    decision: Decision
    expected_loss_if_approved: float
    expected_cost_if_blocked: float
    risk_score: float
    reason: str


def ensemble_risk(p_nn: float, entity_risk: float, anomaly: float, w_nn: float, w_entity: float, w_anomaly: float) -> float:
    total = w_nn + w_entity + w_anomaly
    return float((w_nn * p_nn + w_entity * entity_risk + w_anomaly * anomaly) / total)


def decide(
    p_fraud: float,
    amount: float,
    uncertainty: float,
    anomaly: float,
    entity_risk: float,
    cfg: DecisionConfig,
    weights: tuple[float, float, float] = (0.55, 0.25, 0.20),
) -> DecisionResult:
    risk = ensemble_risk(p_fraud, entity_risk, anomaly, *weights)
    exp_loss_approve = risk * amount * cfg.c_fn_multiplier
    exp_cost_block = (1.0 - risk) * cfg.c_fp

    # Soft boundary from cost equality: risk * amount * m = (1-risk) * c_fp
    # → risk = c_fp / (amount * m + c_fp)
    denom = amount * cfg.c_fn_multiplier + cfg.c_fp
    cost_boundary = cfg.c_fp / denom if denom > 0 else cfg.block_threshold

    if uncertainty >= cfg.review_uncertainty_threshold or abs(risk - cost_boundary) <= cfg.review_margin:
        return DecisionResult(
            decision="REVIEW",
            expected_loss_if_approved=exp_loss_approve,
            expected_cost_if_blocked=exp_cost_block,
            risk_score=risk,
            reason="High uncertainty or near cost boundary — route to analyst",
        )
    if exp_loss_approve > exp_cost_block or risk >= cfg.block_threshold:
        return DecisionResult(
            decision="BLOCK",
            expected_loss_if_approved=exp_loss_approve,
            expected_cost_if_blocked=exp_cost_block,
            risk_score=risk,
            reason="Expected loss of approval exceeds block friction cost",
        )
    return DecisionResult(
        decision="APPROVE",
        expected_loss_if_approved=exp_loss_approve,
        expected_cost_if_blocked=exp_cost_block,
        risk_score=risk,
        reason="Expected approval loss below block cost",
    )
