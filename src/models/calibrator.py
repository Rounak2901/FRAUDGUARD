"""Isotonic probability calibrator wrapper (joblib-safe module path)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Wrap a fitted classifier + isotonic map fitted on validation probabilities."""

    def __init__(self, base, iso: IsotonicRegression) -> None:
        self.base = base
        self.iso = iso

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        cal = np.clip(self.iso.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])
