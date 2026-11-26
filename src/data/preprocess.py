"""
ULB preprocessing: validation, feature transforms, scaler fit on train only, time split.

Outputs under data/processed/:
  train.parquet, val.parquet, test.parquet
  feature_columns.json
  scaler.joblib
  split_summary.json
  meta.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.config import ensure_dirs, load_config, resolve_path, ROOT
from src.data.split import split_summary, time_based_split


FEATURE_V = [f"V{i}" for i in range(1, 29)]


def validate_ulb(df: pd.DataFrame) -> dict[str, Any]:
    """Schema / quality checks. Raises on hard failures."""
    required = ["Time", "Amount", "Class"] + FEATURE_V
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ULB schema missing columns: {missing}")

    n_dup = int(df.duplicated().sum())
    nulls = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": n_dup,
        "null_counts": nulls,
        "fraud_count": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "amount_min": float(df["Amount"].min()),
        "amount_max": float(df["Amount"].max()),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
    }
    if nulls:
        # ULB normally has no nulls; drop if any for robustness
        df.dropna(inplace=True)
        report["dropped_null_rows"] = True
    return report


def add_time_and_amount_features(
    df: pd.DataFrame,
    seconds_per_day: int = 86400,
    amount_log: bool = True,
) -> pd.DataFrame:
    """Cyclical hour-of-day encoding + optional log1p(Amount). Fit-free transforms."""
    out = df.copy()
    hour = (out["Time"] % seconds_per_day) / seconds_per_day * 24.0
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    if amount_log:
        out["Amount_log"] = np.log1p(out["Amount"].astype(float))
    return out


def assign_synthetic_entity_ids(df: pd.DataFrame, seed: int = 42, n_cards: int = 5000, n_merchants: int = 800) -> pd.DataFrame:
    """
    Assign synthetic card_id / merchant_id to ULB (no real IDs in source).

    Limitation: random assignment does not reflect real multi-transaction entity
    behaviour; use PaySim when available for the Bayesian entity module.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["card_id"] = [f"card_{i}" for i in rng.integers(0, n_cards, size=len(out))]
    out["merchant_id"] = [f"merch_{i}" for i in rng.integers(0, n_merchants, size=len(out))]
    out["entity_id"] = out["card_id"]
    return out


def fit_amount_scaler(train: pd.DataFrame, col: str = "Amount_log") -> RobustScaler:
    """Fit RobustScaler on train only — never touch val/test before transform."""
    scaler = RobustScaler()
    scaler.fit(train[[col]].astype(float))
    return scaler


def apply_amount_scaler(df: pd.DataFrame, scaler: RobustScaler, col: str = "Amount_log") -> pd.DataFrame:
    out = df.copy()
    out["Amount_scaled"] = scaler.transform(out[[col]].astype(float)).ravel()
    return out


def model_feature_columns() -> list[str]:
    """Default feature set for supervised models on ULB."""
    return FEATURE_V + ["Amount_scaled", "hour_sin", "hour_cos"]


def preprocess_ulb(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full Phase-1 preprocess pipeline. Returns paths and summary dict."""
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    raw_dir = resolve_path(cfg, "raw_dir")
    proc_dir = resolve_path(cfg, "processed_dir")
    seed = int(cfg["project"]["seed"])

    csv_path = raw_dir / cfg["datasets"]["ulb"]["filename"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run: py -m src.data.download  (or --synthetic)"
        )

    print(f"[preprocess] Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    quality = validate_ulb(df)
    print(
        f"[preprocess] rows={quality['n_rows']} fraud={quality['fraud_count']} "
        f"rate={quality['fraud_rate']:.4%} dups={quality['n_duplicates']}"
    )
    if quality["n_duplicates"] > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    prep_cfg = cfg["preprocessing"]
    df = add_time_and_amount_features(
        df,
        seconds_per_day=int(prep_cfg["seconds_per_day"]),
        amount_log=bool(prep_cfg["amount_log"]),
    )
    df = assign_synthetic_entity_ids(df, seed=seed)

    split_cfg = cfg["split"]
    train, val, test = time_based_split(
        df,
        time_col="Time",
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )

    scaler = fit_amount_scaler(train, col="Amount_log")
    train = apply_amount_scaler(train, scaler)
    val = apply_amount_scaler(val, scaler)
    test = apply_amount_scaler(test, scaler)

    summary = split_summary(train, val, test, target="Class")
    features = model_feature_columns()

    # Persist
    train.to_parquet(proc_dir / "train.parquet", index=False)
    val.to_parquet(proc_dir / "val.parquet", index=False)
    test.to_parquet(proc_dir / "test.parquet", index=False)
    joblib.dump(scaler, proc_dir / "scaler.joblib")
    (proc_dir / "feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (proc_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional: sample PaySim for entity module
    paysim_meta = _maybe_process_paysim(cfg, raw_dir, proc_dir)

    meta = {
        "dataset": "ulb",
        "source_file": str(csv_path.relative_to(ROOT)),
        "quality": quality,
        "split": summary,
        "features": features,
        "entity_ids": "synthetic_ulb (fixed seed) — replace with PaySim for real entity trajectories",
        "paysim": paysim_meta,
        "scaler": "RobustScaler on Amount_log, fit on train only",
        "leakage_controls": [
            "time-based split (no shuffle before cut)",
            "scaler fit on train only",
            "entity rolling features (later) computed causally",
        ],
    }
    (proc_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[preprocess] Wrote train/val/test parquet + scaler + meta")
    for part, st in summary.items():
        print(f"  {part}: n={st['n']} fraud={st['fraud']} rate={st['fraud_rate']:.4%}")
    return meta


def _maybe_process_paysim(cfg: dict, raw_dir: Path, proc_dir: Path) -> dict[str, Any]:
    """Load a PaySim sample (or synthetic) for the entity-risk module."""
    candidates = [
        raw_dir / cfg["datasets"]["paysim"]["filename"],
        raw_dir / "paysim_sample.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {"available": False}

    max_rows = int(cfg["datasets"]["paysim"]["max_rows"])
    seed = int(cfg["datasets"]["paysim"]["sample_seed"])
    print(f"[preprocess] Loading PaySim from {path.name} (max_rows={max_rows}) …")
    df = pd.read_csv(path, nrows=max_rows)
    # Standardise column names used by bayes module
    rename = {}
    if "nameOrig" in df.columns:
        rename["nameOrig"] = "entity_id"
    if "nameDest" in df.columns:
        rename["nameDest"] = "merchant_id"
    if "isFraud" in df.columns:
        rename["isFraud"] = "Class"
    if "amount" in df.columns:
        rename["amount"] = "Amount"
    if "step" in df.columns:
        rename["step"] = "Time"
    df = df.rename(columns=rename)
    if "Class" not in df.columns:
        return {"available": False, "reason": "no fraud label column"}

    # Light sample stratified-ish by keeping all fraud + random genuine
    fraud = df[df["Class"] == 1]
    genuine = df[df["Class"] == 0]
    n_keep = min(len(genuine), max_rows - len(fraud))
    genuine = genuine.sample(n=n_keep, random_state=seed) if n_keep < len(genuine) else genuine
    sample = pd.concat([fraud, genuine], ignore_index=True).sample(frac=1.0, random_state=seed)
    out_path = proc_dir / "paysim_entity.parquet"
    sample.to_parquet(out_path, index=False)
    return {
        "available": True,
        "path": str(out_path.relative_to(ROOT)),
        "n_rows": len(sample),
        "fraud": int(sample["Class"].sum()),
        "n_entities": int(sample["entity_id"].nunique()) if "entity_id" in sample.columns else None,
    }


def main() -> None:
    preprocess_ulb()


if __name__ == "__main__":
    main()
