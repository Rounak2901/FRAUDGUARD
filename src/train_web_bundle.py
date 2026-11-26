"""
Train a laptop-friendly scoring bundle for the web app.

Artifacts → models/web_bundle.joblib
  - random_forest (supervised, class_weight balanced)
  - isolation_forest (anomaly on genuine only)
  - feature columns, metrics, version

Usage: py -m src.train_web_bundle
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import ensure_dirs, load_config, resolve_path, ROOT
from src.evaluation.metrics import binary_metrics, expected_calibration_error, recall_at_precision
from src.models.calibrator import IsotonicCalibrator


def _xy(df: pd.DataFrame, features: list[str]):
    return df[features].astype(float).values, df["Class"].astype(int).values


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    proc = resolve_path(cfg, "processed_dir")
    models_dir = resolve_path(cfg, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    features = json.loads((proc / "feature_columns.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    X_train, y_train = _xy(train, features)
    X_val, y_val = _xy(val, features)
    X_test, y_test = _xy(test, features)

    seed = int(cfg["project"]["seed"])
    print("[train] Logistic Regression …")
    lr = LogisticRegression(
        max_iter=cfg["model"]["baselines"]["logistic_max_iter"],
        class_weight="balanced",
        random_state=seed,
    )
    lr.fit(X_train, y_train)

    print("[train] Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["baselines"]["rf_n_estimators"],
        max_depth=cfg["model"]["baselines"]["rf_max_depth"],
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    print("[train] Calibrating RF (isotonic on val) …")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(rf.predict_proba(X_val)[:, 1], y_val)
    calibrator = IsotonicCalibrator(rf, iso)

    print("[train] Isolation Forest (genuine only) …")
    X_gen = X_train[y_train == 0]
    iforest = IsolationForest(
        n_estimators=cfg["model"]["baselines"]["iforest_n_estimators"],
        contamination=cfg["model"]["baselines"]["iforest_contamination"],
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(X_gen)

    def anomaly_01(X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal → invert & min-max on val
        raw = -iforest.decision_function(X)
        return raw

    raw_val = anomaly_01(X_val)
    a_min, a_max = float(raw_val.min()), float(raw_val.max())
    span = max(a_max - a_min, 1e-9)

    def norm_anomaly(X: np.ndarray) -> np.ndarray:
        return np.clip((anomaly_01(X) - a_min) / span, 0, 1)

    def mc_uncertainty(model: RandomForestClassifier, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tree-wise probability std as epistemic uncertainty proxy (BNN stand-in for web)."""
        all_proba = np.stack([est.predict_proba(X)[:, 1] for est in model.estimators_], axis=0)
        return all_proba.mean(axis=0), all_proba.std(axis=0)

    p_val = calibrator.predict_proba(X_val)[:, 1]
    p_test = calibrator.predict_proba(X_test)[:, 1]
    _, u_test = mc_uncertainty(rf, X_test)

    metrics = {
        "val": binary_metrics(y_val, p_val),
        "test": binary_metrics(y_test, p_test),
        "test_ece": expected_calibration_error(y_test, p_test),
        "test_recall_at_90_precision": recall_at_precision(y_test, p_test, 0.9),
        "lr_test": binary_metrics(y_test, lr.predict_proba(X_test)[:, 1]),
        "rf_uncal_test": binary_metrics(y_test, rf.predict_proba(X_test)[:, 1]),
    }
    print(json.dumps({k: metrics[k] for k in ("val", "test", "lr_test")}, indent=2))

    # Global feature importance for SHAP-like reasons when shap is slow/unavailable
    importances = dict(zip(features, rf.feature_importances_.tolist()))
    background = X_train[np.random.default_rng(seed).choice(len(X_train), size=min(100, len(X_train)), replace=False)]

    bundle = {
        "version": cfg["project"]["version"],
        "features": features,
        "lr": lr,
        "rf": rf,
        "calibrator": calibrator,
        "iforest": iforest,
        "anomaly_min": a_min,
        "anomaly_max": a_max,
        "feature_importances": importances,
        "background_mean": background.mean(axis=0),
        "background_std": background.std(axis=0) + 1e-9,
        "metrics": metrics,
        "ensemble_weights": [
            cfg["ensemble"]["w_nn"],
            cfg["ensemble"]["w_entity"],
            cfg["ensemble"]["w_anomaly"],
        ],
    }
    out = models_dir / "web_bundle.joblib"
    joblib.dump(bundle, out)
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "version": bundle["version"],
                "path": str(out.relative_to(ROOT)),
                "metrics": {"test_pr_auc": metrics["test"]["pr_auc"], "test_roc_auc": metrics["test"]["roc_auc"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results_path = ROOT / cfg["paths"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"web_bundle": metrics}, indent=2), encoding="utf-8")
    print(f"[train] Wrote {out}")


if __name__ == "__main__":
    main()
