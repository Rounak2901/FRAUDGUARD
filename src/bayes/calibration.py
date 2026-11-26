"""Probability calibration helpers and ECE reporting."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.evaluation.metrics import expected_calibration_error


def fit_isotonic(y_prob_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_val, y_val)
    return iso


def apply_isotonic(iso: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(y_prob), 0.0, 1.0)


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> dict:
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "mean_predicted": float(y_prob.mean()),
        "fraud_rate": float(y_true.mean()),
    }
