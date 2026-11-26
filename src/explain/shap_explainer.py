"""SHAP explanations with TreeExplainer surrogate fallback for CPU latency."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.explain.reason_text import top_reasons


class ExplainerService:
    def __init__(self) -> None:
        self.method = "coefficient"
        self._shap = None
        self._tree = None
        self.background = None
        self.feature_names: list[str] = []
        self.lr_coef = None
        self.importances = None

    def fit_surrogate(self, X_bg: np.ndarray, y_bg: np.ndarray, feature_names: list[str], seed: int = 42) -> None:
        self.feature_names = feature_names
        self.background = X_bg
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            import shap

            self._tree = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=80)
            self._tree.fit(X_bg, y_bg)
            self._shap = shap.TreeExplainer(self._tree)
            self.method = "shap_tree_surrogate"
            self.importances = dict(zip(feature_names, self._tree.feature_importances_.tolist()))
        except Exception:
            # Coefficient-style fallback
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
            lr.fit(X_bg, y_bg)
            self.lr_coef = lr.coef_[0]
            self.method = "lr_coefficient"
            self.importances = dict(zip(feature_names, np.abs(self.lr_coef).tolist()))

    def local_reasons(self, x: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        values = dict(zip(self.feature_names, x.tolist()))
        if self._shap is not None:
            sv = self._shap.shap_values(x.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1]
            impacts = dict(zip(self.feature_names, np.asarray(sv).reshape(-1).tolist()))
        elif self.lr_coef is not None:
            mean = self.background.mean(axis=0)
            std = self.background.std(axis=0) + 1e-9
            z = (x - mean) / std
            impacts = dict(zip(self.feature_names, (self.lr_coef * z).tolist()))
        else:
            impacts = {f: 0.0 for f in self.feature_names}
        return top_reasons(impacts, values, k=k)

    def global_importance(self) -> dict[str, float]:
        return self.importances or {}
