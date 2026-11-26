"""
Download public fraud datasets (Kaggle) with manual and synthetic fallbacks.

Dataset roles (documented for the report):
  - ULB creditcard.csv  → supervised classifier, baselines, NN/BNN, SHAP, anomaly
  - PaySim (sample)     → Bayesian entity-risk module (has user/merchant IDs)
  - If PaySim unavailable → synthetic card_id / merchant_id assigned to ULB
    with a fixed seed (limitation stated in docs).

Usage:
  py -m src.data.download
  py -m src.data.download --synthetic   # force synthetic ULB (no Kaggle needed)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ensure_dirs, load_config, resolve_path, ROOT


def _write_readme(raw_dir: Path) -> None:
    readme = raw_dir / "README.md"
    readme.write_text(
        """# Raw data

## ULB Credit Card Fraud (primary)

1. Create a free Kaggle account and accept the dataset terms:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `kaggle.json` in `~/.kaggle/` (Windows: `%USERPROFILE%\\.kaggle\\kaggle.json`).
3. Run: `py -m src.data.download`

**Manual fallback:** download `creditcard.csv` from the Kaggle page above and
copy it to `data/raw/creditcard.csv`.

**Offline / no-Kaggle fallback:** `py -m src.data.download --synthetic`
generates a statistically similar synthetic dataset so the pipeline still runs.

## PaySim (entity Bayesian module)

1. Accept terms: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Run download, or place the CSV as
   `data/raw/PS_20174392719_1491204439457_log.csv`.

If PaySim is missing, the pipeline assigns synthetic `card_id` / `merchant_id`
to ULB rows with seed=42 (documented limitation).
""",
        encoding="utf-8",
    )


def generate_synthetic_ulb(
    n_samples: int = 50000,
    fraud_rate: float = 0.00172,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a ULB-like dataset: Time, V1..V28 (approx. PCA noise), Amount, Class.

    Fraud rows get a shifted feature mean and higher amounts so models learn signal.
    This is for local demos when Kaggle data is unavailable — not for published scores.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_samples * fraud_rate), 50)
    n_gen = n_samples - n_fraud

    def _block(n: int, fraud: bool) -> pd.DataFrame:
        time = rng.uniform(0, 172792, size=n)  # ~2 days like ULB
        # PCA-like features
        V = rng.normal(0, 1, size=(n, 28))
        if fraud:
            # Shift a few informative dimensions
            V[:, 13] += rng.normal(-3.5, 1.0, size=n)  # V14-like
            V[:, 11] += rng.normal(-2.0, 1.0, size=n)
            V[:, 16] += rng.normal(2.5, 1.0, size=n)
            amount = rng.lognormal(mean=5.0, sigma=1.2, size=n)
        else:
            amount = rng.lognormal(mean=3.5, sigma=1.0, size=n)
        cols = {f"V{i}": V[:, i - 1] for i in range(1, 29)}
        cols["Time"] = time
        cols["Amount"] = amount
        cols["Class"] = 1 if fraud else 0
        return pd.DataFrame(cols)

    df = pd.concat([_block(n_gen, False), _block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Reorder columns like ULB
    ordered = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    return df[ordered]


def generate_synthetic_paysim(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Minimal PaySim-like table with nameOrig / nameDest for entity Bayes demos."""
    rng = np.random.default_rng(seed)
    types = np.array(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    is_fraud = rng.random(n_samples) < 0.0015
    # Fraud more common on TRANSFER / CASH_OUT
    tx_type = rng.choice(types, size=n_samples, p=[0.35, 0.2, 0.25, 0.1, 0.1])
    tx_type = np.where(is_fraud & (rng.random(n_samples) < 0.8), "TRANSFER", tx_type)

    n_users = 5000
    n_merchants = 800
    amount = rng.lognormal(4.0, 1.1, size=n_samples)
    amount = np.where(is_fraud, amount * rng.uniform(2, 8, size=n_samples), amount)

    df = pd.DataFrame(
        {
            "step": rng.integers(1, 400, size=n_samples),
            "type": tx_type,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in rng.integers(0, n_users, size=n_samples)],
            "oldbalanceOrg": rng.uniform(0, 1e5, size=n_samples),
            "newbalanceOrig": rng.uniform(0, 1e5, size=n_samples),
            "nameDest": [f"M{i}" for i in rng.integers(0, n_merchants, size=n_samples)],
            "oldbalanceDest": rng.uniform(0, 1e5, size=n_samples),
            "newbalanceDest": rng.uniform(0, 1e5, size=n_samples),
            "isFraud": is_fraud.astype(int),
            "isFlaggedFraud": 0,
        }
    )
    return df


def _try_kaggle_download(dataset: str, dest_dir: Path) -> bool:
    """Attempt Kaggle CLI download. Returns True on success."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"[kaggle] Downloading {dataset} → {dest_dir}")
        api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)
        return True
    except Exception as exc:  # noqa: BLE001 — intentional soft failure
        print(f"[kaggle] Failed for {dataset}: {exc}")
        return False


def _ensure_ulb(raw_dir: Path, force_synthetic: bool, seed: int) -> Path:
    target = raw_dir / "creditcard.csv"
    if target.exists() and not force_synthetic:
        print(f"[ulb] Found existing {target}")
        return target

    if not force_synthetic:
        ok = _try_kaggle_download("mlg-ulb/creditcardfraud", raw_dir)
        if ok and target.exists():
            return target
        # Sometimes zip left behind
        for z in raw_dir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(raw_dir)
            if target.exists():
                return target

    print("[ulb] Using synthetic ULB-like dataset (Kaggle unavailable or --synthetic).")
    print("      Decision: synthetic data for local runnable demos; replace with real")
    print("      creditcard.csv for thesis-quality metrics.")
    df = generate_synthetic_ulb(n_samples=50000, seed=seed)
    df.to_csv(target, index=False)
    meta = raw_dir / "creditcard.SOURCE.txt"
    meta.write_text(
        "SYNTHETIC — generated by src.data.download.generate_synthetic_ulb\n"
        f"sha256_of_generator_note={hashlib.sha256(b'synthetic_ulb_v1').hexdigest()}\n",
        encoding="utf-8",
    )
    print(f"[ulb] Wrote {target} shape={df.shape} fraud={df['Class'].sum()}")
    return target


def _ensure_paysim(raw_dir: Path, cfg: dict, force_synthetic: bool, seed: int) -> Path | None:
    fname = cfg["datasets"]["paysim"]["filename"]
    target = raw_dir / fname
    syn_target = raw_dir / "paysim_sample.csv"

    if force_synthetic:
        df = generate_synthetic_paysim(n_samples=min(50000, cfg["datasets"]["paysim"]["max_rows"]), seed=seed)
        df.to_csv(syn_target, index=False)
        print(f"[paysim] Wrote synthetic sample {syn_target} shape={df.shape}")
        return syn_target

    if target.exists():
        print(f"[paysim] Found existing {target}")
        return target
    if syn_target.exists():
        print(f"[paysim] Found existing {syn_target}")
        return syn_target

    ok = _try_kaggle_download(cfg["datasets"]["paysim"]["kaggle_dataset"], raw_dir)
    if ok and target.exists():
        return target

    print("[paysim] Not available — will use synthetic entity IDs on ULB for Bayes module.")
    # Still write a small synthetic PaySim so entity demos work out of the box
    df = generate_synthetic_paysim(n_samples=50000, seed=seed)
    df.to_csv(syn_target, index=False)
    print(f"[paysim] Wrote fallback synthetic {syn_target}")
    return syn_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or synthesise fraud datasets")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data (no Kaggle)")
    parser.add_argument("--skip-paysim", action="store_true", help="Skip PaySim download")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    raw_dir = resolve_path(cfg, "raw_dir")
    _write_readme(raw_dir)

    seed = int(cfg["project"]["seed"])
    ulb_path = _ensure_ulb(raw_dir, force_synthetic=args.synthetic, seed=seed)
    print(f"[done] Primary dataset: {ulb_path}")

    if not args.skip_paysim:
        paysim_path = _ensure_paysim(raw_dir, cfg, force_synthetic=args.synthetic, seed=seed)
        print(f"[done] Entity dataset: {paysim_path}")

    # Marker for processed pipeline
    (raw_dir / ".download_complete").touch()
    print("[done] Phase 1 data ready. Next: py -m src.data.eda && py -m src.data.preprocess")


if __name__ == "__main__":
    main()
