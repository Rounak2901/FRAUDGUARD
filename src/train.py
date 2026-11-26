"""
One-command full training pipeline.

  py -m src.train

Trains baselines, MLP (BCE weighted + focal), MC-Dropout BNN, autoencoder,
ensemble weights, cost thresholds, explainer — writes models/ + reports/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.bayes.calibration import apply_isotonic, calibration_report, fit_isotonic
from src.bayes.entity_risk import EntityRiskStore
from src.config import ensure_dirs, load_config, resolve_path, ROOT
from src.decision.threshold import optimize_threshold
from src.evaluation.ablation import run_ablation
from src.evaluation.metrics import binary_metrics, expected_calibration_error, recall_at_precision
from src.evaluation.plots import save_confusion, save_reliability, save_roc_pr, save_uncertainty_vs_error
from src.explain.shap_explainer import ExplainerService
from src.models.autoencoder import normalize_scores, reconstruction_error, train_autoencoder
from src.models.baselines import train_baselines
from src.models.bnn import mc_dropout_predict
from src.models.ensemble import combine, grid_search_weights
from src.models.nn import predict_proba, train_mlp


def _xy(df: pd.DataFrame, features: list[str]):
    return df[features].astype(float).values, df["Class"].astype(int).values


def _entity_risk_vector(df: pd.DataFrame, store: EntityRiskStore) -> np.ndarray:
    """Causal soft entity prior: peek without label update (prior only unless history exists)."""
    # Use synthetic card entity posterior mean after walking chronologically with true labels
    # (offline simulation of feedback) — causal: update AFTER scoring the row.
    risks = []
    ordered = df.sort_values("Time")
    tmp = EntityRiskStore(alpha0=store.alpha0, beta0=store.beta0, decay_per_day=store.decay_per_day)
    for _, r in ordered.iterrows():
        eid = str(r.get("entity_id", r.get("card_id", "unknown")))
        risks.append(tmp.peek(eid)["mean"])
        tmp.update(eid, is_fraud=bool(r["Class"]))
    # Reindex to original order
    out = pd.Series(risks, index=ordered.index).reindex(df.index).values
    return out.astype(float)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    proc = resolve_path(cfg, "processed_dir")
    models_dir = resolve_path(cfg, "models_dir")
    fig_dir = resolve_path(cfg, "figures_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    features = json.loads((proc / "feature_columns.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    X_tr, y_tr = _xy(train, features)
    X_va, y_va = _xy(val, features)
    X_te, y_te = _xy(test, features)
    amt_va = val["Amount"].values.astype(float)
    amt_te = test["Amount"].values.astype(float)

    results: dict = {"version": cfg["project"]["version"], "seed": seed}

    print("=" * 60)
    print("[1/7] Baselines")
    results["baselines"] = train_baselines()

    print("=" * 60)
    print("[2/7] Neural nets (weighted BCE vs focal)")
    nn_cfg = cfg["model"]["nn"]
    mlp_bce = train_mlp(
        X_tr, y_tr, X_va, y_va,
        hidden_dims=tuple(nn_cfg["hidden_dims"]),
        dropout=nn_cfg["dropout"],
        epochs=nn_cfg["epochs"],
        batch_size=nn_cfg["batch_size"],
        lr=nn_cfg["lr"],
        weight_decay=nn_cfg["weight_decay"],
        patience=nn_cfg["early_stopping_patience"],
        seed=seed,
        loss_type="bce_weighted",
    )
    mlp_focal = train_mlp(
        X_tr, y_tr, X_va, y_va,
        hidden_dims=tuple(nn_cfg["hidden_dims"]),
        dropout=nn_cfg["dropout"],
        epochs=nn_cfg["epochs"],
        batch_size=nn_cfg["batch_size"],
        lr=nn_cfg["lr"],
        weight_decay=nn_cfg["weight_decay"],
        patience=nn_cfg["early_stopping_patience"],
        seed=seed,
        loss_type="focal",
        focal_gamma=nn_cfg["focal_gamma"],
        focal_alpha=nn_cfg["focal_alpha"],
    )
    # Pick better by val PR-AUC
    mlp = mlp_bce if mlp_bce.best_val_pr_auc >= mlp_focal.best_val_pr_auc else mlp_focal
    chosen_loss = "bce_weighted" if mlp is mlp_bce else "focal"
    torch.save(mlp.model.state_dict(), models_dir / "nn_mlp.pt")
    p_va = predict_proba(mlp.model, X_va)
    p_te = predict_proba(mlp.model, X_te)
    results["nn"] = {
        "chosen_loss": chosen_loss,
        "val_pr_auc_bce": mlp_bce.best_val_pr_auc,
        "val_pr_auc_focal": mlp_focal.best_val_pr_auc,
        "val": binary_metrics(y_va, p_va),
        "test": binary_metrics(y_te, p_te),
    }
    print(f"  chosen={chosen_loss} test PR-AUC={results['nn']['test']['pr_auc']:.4f}")

    print("=" * 60)
    print("[3/7] MC Dropout BNN + calibration")
    bnn_cfg = cfg["model"]["bnn"]
    mc_va = mc_dropout_predict(mlp.model, X_va, n_passes=int(bnn_cfg["mc_passes"]))
    mc_te = mc_dropout_predict(mlp.model, X_te, n_passes=int(bnn_cfg["mc_passes"]))
    iso = fit_isotonic(mc_va["p_fraud_mean"], y_va)
    p_va_cal = apply_isotonic(iso, mc_va["p_fraud_mean"])
    p_te_cal = apply_isotonic(iso, mc_te["p_fraud_mean"])
    joblib.dump(iso, models_dir / "isotonic.joblib")
    results["bnn"] = {
        "method": "mc_dropout",
        "mc_passes": bnn_cfg["mc_passes"],
        "test_uncal": binary_metrics(y_te, mc_te["p_fraud_mean"]),
        "test_cal": binary_metrics(y_te, p_te_cal),
        "ece_before": expected_calibration_error(y_te, mc_te["p_fraud_mean"]),
        "ece_after": expected_calibration_error(y_te, p_te_cal),
        "calibration": calibration_report(y_te, p_te_cal),
        "recall_at_90_precision": recall_at_precision(y_te, p_te_cal, 0.9),
        "mean_uncertainty_correct": float(mc_te["uncertainty_std"][(p_te_cal >= 0.5).astype(int) == y_te].mean()),
        "mean_uncertainty_errors": float(mc_te["uncertainty_std"][(p_te_cal >= 0.5).astype(int) != y_te].mean())
        if ((p_te_cal >= 0.5).astype(int) != y_te).any()
        else 0.0,
    }
    print(f"  ECE {results['bnn']['ece_before']:.4f} → {results['bnn']['ece_after']:.4f}")

    print("=" * 60)
    print("[4/7] Autoencoder anomaly")
    ae_cfg = cfg["model"]["autoencoder"]
    ae = train_autoencoder(
        X_tr[y_tr == 0],
        hidden_dims=tuple(ae_cfg["hidden_dims"]),
        epochs=ae_cfg["epochs"],
        batch_size=ae_cfg["batch_size"],
        lr=ae_cfg["lr"],
        seed=seed,
    )
    torch.save(ae.state_dict(), models_dir / "autoencoder.pt")
    err_va = reconstruction_error(ae, X_va)
    err_te = reconstruction_error(ae, X_te)
    a_min, a_max = float(err_va.min()), float(err_va.max())
    anom_va = normalize_scores(err_va, a_min, a_max)
    anom_te = normalize_scores(err_te, a_min, a_max)
    results["autoencoder"] = {
        "test": binary_metrics(y_te, anom_te),
        "anomaly_min": a_min,
        "anomaly_max": a_max,
    }

    print("=" * 60)
    print("[5/7] Entity Bayes features + ensemble")
    store = EntityRiskStore(
        alpha0=float(cfg["bayes_entity"]["alpha0"]),
        beta0=float(cfg["bayes_entity"]["beta0"]),
        decay_per_day=float(cfg["bayes_entity"]["decay_per_day"]),
    )
    ent_va = _entity_risk_vector(val, store)
    ent_te = _entity_risk_vector(test, store)
    weights = grid_search_weights(p_va_cal, ent_va, anom_va, y_va)
    ens_va = combine(p_va_cal, ent_va, anom_va, weights)
    ens_te = combine(p_te_cal, ent_te, anom_te, weights)
    (models_dir / "ensemble_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    results["ensemble"] = {"weights": weights, "val": binary_metrics(y_va, ens_va), "test": binary_metrics(y_te, ens_te)}
    print(f"  weights={weights} test PR-AUC={results['ensemble']['test']['pr_auc']:.4f}")

    print("=" * 60)
    print("[6/7] Cost-sensitive threshold + ablation")
    thr = optimize_threshold(
        y_va, ens_va, amt_va,
        c_fp=float(cfg["decision"]["c_fp"]),
        c_fn_multiplier=float(cfg["decision"]["c_fn_multiplier"]),
    )
    (models_dir / "decision_thresholds.json").write_text(json.dumps({
        "block_threshold": thr["optimal_threshold"],
        "optimal_cost_val": thr["optimal_cost"],
        "default_0_5_cost_val": thr["default_0_5_cost"],
    }, indent=2), encoding="utf-8")
    test_cost_opt = optimize_threshold(
        y_te, ens_te, amt_te,
        c_fp=float(cfg["decision"]["c_fp"]),
        c_fn_multiplier=float(cfg["decision"]["c_fn_multiplier"]),
        grid=[thr["optimal_threshold"]],
    )
    results["decision"] = {
        "val_optimization": {k: v for k, v in thr.items() if k != "curve"},
        "test_cost_at_val_threshold": test_cost_opt["optimal_cost"],
        "test_cost_at_0_5": optimize_threshold(y_te, ens_te, amt_te, grid=[0.5])["optimal_cost"],
    }
    ablation = run_ablation(
        y_te,
        {
            "nn_only": p_te_cal,
            "nn_plus_entity": 0.7 * p_te_cal + 0.3 * ent_te,
            "nn_plus_anomaly": 0.7 * p_te_cal + 0.3 * anom_te,
            "full_ensemble": ens_te,
        },
        amt_te,
        c_fp=float(cfg["decision"]["c_fp"]),
    )
    results["ablation"] = ablation

    print("=" * 60)
    print("[7/7] Explainer + figures + registry")
    explainer = ExplainerService()
    bg_idx = np.random.default_rng(seed).choice(len(X_tr), size=min(200, len(X_tr)), replace=False)
    explainer.fit_surrogate(X_tr[bg_idx], y_tr[bg_idx], features, seed=seed)
    joblib.dump(explainer, models_dir / "shap_explainer.joblib")
    results["explain_method"] = explainer.method
    results["global_importance"] = dict(
        sorted(explainer.global_importance().items(), key=lambda kv: kv[1], reverse=True)[:15]
    )

    # Figures
    save_roc_pr(y_te, ens_te, fig_dir, prefix="ensemble")
    save_roc_pr(y_te, p_te_cal, fig_dir, prefix="bnn")
    save_confusion(y_te, (ens_te >= thr["optimal_threshold"]).astype(int), fig_dir, prefix="ensemble")
    save_reliability(y_te, p_te_cal, fig_dir, prefix="bnn")
    save_uncertainty_vs_error(y_te, p_te_cal, mc_te["uncertainty_std"], fig_dir, prefix="bnn")

    # Meta for API / web bundle compatibility
    meta = {
        "n_features": len(features),
        "features": features,
        "hidden_dims": list(nn_cfg["hidden_dims"]),
        "dropout": nn_cfg["dropout"],
        "anomaly_min": a_min,
        "anomaly_max": a_max,
        "ensemble_weights": weights,
        "block_threshold": thr["optimal_threshold"],
        "chosen_loss": chosen_loss,
    }
    (models_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "version": cfg["project"]["version"],
                "trained_artifacts": [
                    "nn_mlp.pt",
                    "autoencoder.pt",
                    "isotonic.joblib",
                    "ensemble_weights.json",
                    "decision_thresholds.json",
                    "shap_explainer.joblib",
                    "baselines/",
                    "web_bundle.joblib",
                ],
                "test_pr_auc_ensemble": results["ensemble"]["test"]["pr_auc"],
                "test_pr_auc_bnn": results["bnn"]["test_cal"]["pr_auc"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results_path = ROOT / cfg["paths"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"[done] Wrote {results_path}")
    print(f"[done] Ensemble test PR-AUC={results['ensemble']['test']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
