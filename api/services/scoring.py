"""Load model bundle and score transactions end-to-end."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from scipy.stats import norm

from api.schemas import ScoreOut, TransactionIn
from src.bayes.entity_risk import EntityRiskStore
from src.config import load_config, resolve_path, ROOT
from src.data.features import OnlineFeatureBuilder
from src.data.preprocess import add_time_and_amount_features, FEATURE_V
from src.decision.engine import DecisionConfig, decide
from src.explain.reason_text import top_reasons


class ScoringService:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.bundle: dict[str, Any] | None = None
        self.scaler = None
        self.entity_store = EntityRiskStore(
            alpha0=float(self.cfg["bayes_entity"]["alpha0"]),
            beta0=float(self.cfg["bayes_entity"]["beta0"]),
            decay_per_day=float(self.cfg["bayes_entity"]["decay_per_day"]),
        )
        self.feature_builder = OnlineFeatureBuilder()
        self.version = self.cfg["project"]["version"]
        self.decision_cfg = DecisionConfig(
            c_fp=float(self.cfg["decision"]["c_fp"]),
            c_fn_multiplier=float(self.cfg["decision"]["c_fn_multiplier"]),
            c_review=float(self.cfg["decision"]["c_review"]),
            review_uncertainty_threshold=float(self.cfg["decision"]["review_uncertainty_threshold"]),
            review_margin=float(self.cfg["decision"]["review_margin"]),
            block_threshold=float(self.cfg["decision"]["block_threshold"]),
        )
        self.weights = tuple(self.cfg["ensemble"][k] for k in ("w_nn", "w_entity", "w_anomaly"))
        self.metrics_live = {
            "scored": 0,
            "blocked": 0,
            "reviewed": 0,
            "approved": 0,
            "fraud_labeled": 0,
            "latency_sum_ms": 0.0,
            "amount_blocked": 0.0,
            "cost_saved_est": 0.0,
        }

    def load(self) -> None:
        models_dir = resolve_path(self.cfg, "models_dir")
        bundle_path = models_dir / "web_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Missing {bundle_path}. Run: py -m src.train_web_bundle"
            )
        self.bundle = joblib.load(bundle_path)
        self.version = self.bundle.get("version", self.version)
        scaler_path = resolve_path(self.cfg, "processed_dir") / "scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        print(f"[scoring] Loaded bundle v{self.version} from {bundle_path}")

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def _vectorize(self, tx: TransactionIn) -> tuple[np.ndarray, dict[str, float]]:
        assert self.bundle and self.scaler
        feats = dict(tx.features or {})
        row = {"Time": float(tx.time), "Amount": float(tx.amount), "Class": 0}
        for v in FEATURE_V:
            row[v] = float(feats.get(v, 0.0))
        import pandas as pd

        df = add_time_and_amount_features(pd.DataFrame([row]))
        df["Amount_scaled"] = self.scaler.transform(df[["Amount_log"]].astype(float)).ravel()
        columns = self.bundle["features"]
        vec = df[columns].astype(float).values[0]
        values = {c: float(df[c].iloc[0]) if c in df.columns else float(feats.get(c, 0.0)) for c in columns}
        return vec, values

    def _anomaly(self, x: np.ndarray) -> float:
        assert self.bundle
        raw = float(-self.bundle["iforest"].decision_function(x.reshape(1, -1))[0])
        span = max(self.bundle["anomaly_max"] - self.bundle["anomaly_min"], 1e-9)
        return float(np.clip((raw - self.bundle["anomaly_min"]) / span, 0, 1))

    def _mc_proba(self, x: np.ndarray) -> tuple[float, float, list[float]]:
        """Calibrated mean + epistemic std from a subset of RF trees (CPU-friendly)."""
        assert self.bundle
        rf = self.bundle["rf"]
        x2 = x.reshape(1, -1)
        p_cal = float(self.bundle["calibrator"].predict_proba(x2)[0, 1])
        # Sample up to 20 trees instead of all estimators (keeps p95 latency low on CPU)
        n = len(rf.estimators_)
        t = min(20, n)
        rng = np.random.default_rng(abs(hash(x.tobytes())) % (2**32))
        idx = rng.choice(n, size=t, replace=False)
        tree_p = np.fromiter(
            (rf.estimators_[i].predict_proba(x2)[0, 1] for i in idx),
            dtype=float,
            count=t,
        )
        std = float(tree_p.std())
        lo = float(np.clip(norm.ppf(0.025, loc=p_cal, scale=max(std, 1e-6)), 0, 1))
        hi = float(np.clip(norm.ppf(0.975, loc=p_cal, scale=max(std, 1e-6)), 0, 1))
        return p_cal, std, [lo, hi]

    def _local_impacts(self, x: np.ndarray, values: dict[str, float]) -> dict[str, float]:
        """Fast local explanation: importance × standardized deviation from background."""
        assert self.bundle
        mean = self.bundle["background_mean"]
        std = self.bundle["background_std"]
        imps = self.bundle["feature_importances"]
        features = self.bundle["features"]
        z = (x - mean) / std
        # Sign: positive z on important fraud-leaning features → toward fraud
        # Use calibrated coefficient proxy from LR when available
        lr_coef = getattr(self.bundle["lr"], "coef_", None)
        impacts = {}
        for i, f in enumerate(features):
            base = float(imps.get(f, 0.0)) * float(z[i])
            if lr_coef is not None:
                base = float(lr_coef[0, i]) * float(z[i]) * 0.15 + base * 0.05
            impacts[f] = base
        return impacts

    def score(self, tx: TransactionIn, persist_hooks: Optional[dict] = None) -> ScoreOut:
        if not self.ready:
            raise RuntimeError("Scoring bundle not loaded")
        t0 = time.perf_counter()
        tx_id = tx.transaction_id or f"tx_{uuid.uuid4().hex[:10]}"
        x, values = self._vectorize(tx)

        # Entity rolling features (inform reasons; model uses ULB PCA set)
        ent_feat = self.feature_builder.transform_one(
            tx.entity_id, timestamp=float(tx.time), amount=float(tx.amount), merchant_id=tx.merchant_id
        )
        p_fraud, unc, ci = self._mc_proba(x)
        anomaly = self._anomaly(x)
        entity_peek = self.entity_store.peek(tx.entity_id)
        entity_risk = float(entity_peek["mean"])

        # Soft-update entity belief with model soft evidence (weak) so live chart moves;
        # hard update happens on analyst feedback.
        # Skip soft update to avoid double-counting — only feedback updates α/β.

        decision = decide(
            p_fraud=p_fraud,
            amount=float(tx.amount),
            uncertainty=unc,
            anomaly=anomaly,
            entity_risk=entity_risk,
            cfg=self.decision_cfg,
            weights=self.weights,  # type: ignore[arg-type]
        )
        impacts = self._local_impacts(x, values)
        impacts["entity_risk"] = entity_risk * 0.2
        impacts["anomaly_score"] = (anomaly - 0.5) * 0.3
        if ent_feat["amount_zscore"]:
            impacts["amount_zscore"] = float(ent_feat["amount_zscore"]) * 0.05
        reasons = top_reasons(impacts, {**values, "amount_zscore": ent_feat["amount_zscore"]}, k=int(self.cfg["explain"]["top_k_reasons"]))

        latency = (time.perf_counter() - t0) * 1000.0
        self._update_live_kpis(decision.decision, float(tx.amount), latency, decision.expected_loss_if_approved)

        out = ScoreOut(
            transaction_id=tx_id,
            p_fraud=round(p_fraud, 6),
            uncertainty_std=round(unc, 6),
            credible_interval_95=[round(ci[0], 6), round(ci[1], 6)],
            anomaly_score=round(anomaly, 6),
            entity_risk_posterior=round(entity_risk, 6),
            risk_score=round(decision.risk_score, 6),
            decision=decision.decision,
            expected_loss_if_approved=round(decision.expected_loss_if_approved, 4),
            top_reasons=reasons,
            latency_ms=round(latency, 3),
            model_version=self.version,
            amount=float(tx.amount),
            entity_id=tx.entity_id,
            merchant_id=tx.merchant_id,
        )
        if persist_hooks and "save" in persist_hooks:
            persist_hooks["save"](tx, out, values)
        return out

    def _update_live_kpis(self, decision: str, amount: float, latency: float, exp_loss: float) -> None:
        m = self.metrics_live
        m["scored"] += 1
        m["latency_sum_ms"] += latency
        key = {"BLOCK": "blocked", "REVIEW": "reviewed", "APPROVE": "approved"}[decision]
        m[key] += 1
        if decision == "BLOCK":
            m["amount_blocked"] += amount
            m["cost_saved_est"] += max(0.0, exp_loss - self.decision_cfg.c_fp)

    def apply_feedback(self, entity_id: str, is_fraud: bool) -> dict:
        b = self.entity_store.update(entity_id, is_fraud=is_fraud)
        if is_fraud:
            self.metrics_live["fraud_labeled"] += 1
        return self.entity_store.peek(entity_id)


# Singleton used by API
scoring_service = ScoringService()
