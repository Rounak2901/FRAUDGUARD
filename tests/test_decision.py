from src.decision.engine import DecisionConfig, decide
from src.decision.threshold import optimize_threshold, total_cost
import numpy as np


def test_high_prob_large_amount_blocks():
    cfg = DecisionConfig(c_fp=5.0, review_uncertainty_threshold=0.5, review_margin=0.0, block_threshold=0.9)
    r = decide(0.95, amount=1000, uncertainty=0.01, anomaly=0.9, entity_risk=0.2, cfg=cfg)
    assert r.decision == "BLOCK"


def test_low_risk_approves():
    cfg = DecisionConfig(c_fp=5.0, review_uncertainty_threshold=0.5, review_margin=0.0, block_threshold=0.9)
    r = decide(0.01, amount=10, uncertainty=0.01, anomaly=0.05, entity_risk=0.002, cfg=cfg)
    assert r.decision == "APPROVE"


def test_high_uncertainty_reviews():
    cfg = DecisionConfig(review_uncertainty_threshold=0.1, review_margin=0.0)
    r = decide(0.2, amount=50, uncertainty=0.3, anomaly=0.1, entity_risk=0.01, cfg=cfg)
    assert r.decision == "REVIEW"


def test_optimize_threshold_beats_or_equals_default():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    p = np.clip(y * 0.8 + rng.random(100) * 0.3, 0, 1)
    amt = rng.uniform(10, 200, size=100)
    out = optimize_threshold(y, p, amt, c_fp=5.0)
    assert out["optimal_cost"] <= out["default_0_5_cost"] + 1e-9
    assert total_cost(y, p, amt, out["optimal_threshold"], 5.0) == out["optimal_cost"]
