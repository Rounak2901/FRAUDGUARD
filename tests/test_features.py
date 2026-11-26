"""Causal entity feature builder tests."""

from src.data.features import OnlineFeatureBuilder


def test_first_transaction_has_empty_history_features():
    b = OnlineFeatureBuilder()
    f = b.transform_one("card_1", timestamp=100.0, amount=50.0, merchant_id="m1")
    assert f["tx_count_1h"] == 0
    assert f["time_since_last"] == -1.0
    assert f["first_time_merchant"] == 1.0


def test_second_transaction_sees_first_only():
    b = OnlineFeatureBuilder()
    b.transform_one("card_1", timestamp=100.0, amount=50.0, merchant_id="m1")
    f = b.transform_one("card_1", timestamp=200.0, amount=150.0, merchant_id="m1")
    assert f["tx_count_1h"] == 1
    assert f["time_since_last"] == 100.0
    assert f["first_time_merchant"] == 0.0
    assert f["amount_mean"] == 50.0


def test_entities_are_isolated():
    b = OnlineFeatureBuilder()
    b.transform_one("card_a", timestamp=1.0, amount=10.0, merchant_id="m")
    f = b.transform_one("card_b", timestamp=2.0, amount=99.0, merchant_id="m")
    assert f["tx_count_24h"] == 0
