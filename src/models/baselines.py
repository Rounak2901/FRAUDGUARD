"""Classical baselines: Logistic Regression, Random Forest, XGBoost (optional), Isolation Forest."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ensure_dirs, load_config, resolve_path, ROOT
from src.evaluation.metrics import binary_metrics


def _try_xgb(**kwargs):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=kwargs.get("seed", 42),
            n_jobs=-1,
            scale_pos_weight=kwargs.get("scale_pos_weight", 1.0),
        )
    except Exception:
        return None


def train_baselines() -> dict:
    cfg = load_config()
    ensure_dirs(cfg)
    proc = resolve_path(cfg, "processed_dir")
    out_dir = resolve_path(cfg, "models_dir") / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    features = json.loads((proc / "feature_columns.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    X_tr, y_tr = train[features].values, train["Class"].values
    X_te, y_te = test[features].values, test["Class"].values
    seed = int(cfg["project"]["seed"])
    bcfg = cfg["model"]["baselines"]

    models = {}
    lr = LogisticRegression(max_iter=bcfg["logistic_max_iter"], class_weight="balanced", random_state=seed)
    lr.fit(X_tr, y_tr)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=bcfg["rf_n_estimators"],
        max_depth=bcfg["rf_max_depth"],
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    models["random_forest"] = rf

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb = _try_xgb(
        n_estimators=bcfg["xgb_n_estimators"],
        max_depth=bcfg["xgb_max_depth"],
        learning_rate=bcfg["xgb_learning_rate"],
        seed=seed,
        scale_pos_weight=spw,
    )
    if xgb is not None:
        xgb.fit(X_tr, y_tr)
        models["xgboost"] = xgb

    iforest = IsolationForest(
        n_estimators=bcfg["iforest_n_estimators"],
        contamination=bcfg["iforest_contamination"],
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(X_tr[y_tr == 0])
    models["isolation_forest"] = iforest

    results = {}
    for name, model in models.items():
        if name == "isolation_forest":
            raw = -model.decision_function(X_te)
            p = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
        else:
            p = model.predict_proba(X_te)[:, 1]
        results[name] = binary_metrics(y_te, p)
        joblib.dump(model, out_dir / f"{name}.joblib")
        print(f"[baselines] {name} PR-AUC={results[name]['pr_auc']:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    train_baselines()


if __name__ == "__main__":
    main()
