"""Unit tests for time-based split (no leakage of future into train)."""

import pandas as pd
import pytest

from src.data.split import time_based_split


def test_time_based_split_order_and_sizes():
    df = pd.DataFrame(
        {
            "Time": [10, 1, 5, 20, 15, 3, 8, 12, 18, 25],
            "Class": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            "Amount": range(10),
        }
    )
    train, val, test = time_based_split(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3)
    assert len(train) == 5
    assert len(val) == 2
    assert len(test) == 3
    assert train["Time"].is_monotonic_increasing
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_split_ratio_validation():
    df = pd.DataFrame({"Time": [1, 2, 3], "Class": [0, 1, 0]})
    with pytest.raises(ValueError):
        time_based_split(df, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
