from src.bayes.entity_risk import EntityRiskStore


def test_prior_mean_near_global_rate():
    s = EntityRiskStore(alpha0=1.0, beta0=500.0)
    b = s.peek("card_x")
    assert abs(b["mean"] - 1 / 501) < 1e-9


def test_fraud_increases_mean():
    s = EntityRiskStore(alpha0=1.0, beta0=500.0)
    before = s.peek("c1")["mean"]
    s.update("c1", is_fraud=True)
    after = s.peek("c1")["mean"]
    assert after > before
    assert len(s.history("c1")) == 1


def test_genuine_decreases_mean():
    s = EntityRiskStore(alpha0=1.0, beta0=500.0)
    s.update("c2", is_fraud=True)
    mid = s.peek("c2")["mean"]
    s.update("c2", is_fraud=False)
    assert s.peek("c2")["mean"] < mid
