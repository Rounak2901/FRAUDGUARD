"""Data loading, download, preprocessing, feature engineering, and splits."""

__all__ = ["time_based_split", "preprocess_ulb"]


def __getattr__(name: str):
    if name == "time_based_split":
        from .split import time_based_split

        return time_based_split
    if name == "preprocess_ulb":
        from .preprocess import preprocess_ulb

        return preprocess_ulb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
