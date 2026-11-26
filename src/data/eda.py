"""
Exploratory data analysis for the ULB (or synthetic) credit-card dataset.

Saves figures to reports/figures/ at 300 dpi and writes a short eda_summary.json.
Also regenerates notebooks/01_eda.ipynb as a thin wrapper for interactive use.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ensure_dirs, load_config, resolve_path, ROOT

sns.set_theme(style="whitegrid", context="notebook")
DPI = 300


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def run_eda(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    raw_dir = resolve_path(cfg, "raw_dir")
    fig_dir = resolve_path(cfg, "figures_dir")
    csv_path = raw_dir / cfg["datasets"]["ulb"]["filename"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run py -m src.data.download first.")

    print(f"[eda] Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    feature_v = [f"V{i}" for i in range(1, 29)]

    summary = {
        "n_rows": int(len(df)),
        "n_fraud": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "amount_describe": df["Amount"].describe().to_dict(),
        "time_range": [float(df["Time"].min()), float(df["Time"].max())],
    }

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(["Genuine (0)", "Fraud (1)"], counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_ylabel("Count")
    ax.set_title("Class distribution (extreme imbalance)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df):.3%})", ha="center", va="bottom", fontsize=9)
    _savefig(fig, fig_dir / "01_class_distribution.png")

    # 2. Amount distribution fraud vs genuine (log)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [(0, "#2a9d8f"), (1, "#e76f51")]:
        subset = df.loc[df["Class"] == label, "Amount"]
        ax.hist(np.log1p(subset), bins=50, alpha=0.55, label="Genuine" if label == 0 else "Fraud", color=color, density=True)
    ax.set_xlabel("log(1 + Amount) [Amount in dataset currency units]")
    ax.set_ylabel("Density")
    ax.set_title("Amount distribution: fraud vs genuine")
    ax.legend()
    _savefig(fig, fig_dir / "02_amount_log_by_class.png")

    # 3. Time distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df.loc[df["Class"] == 0, "Time"], bins=80, alpha=0.5, label="Genuine", color="#2a9d8f", density=True)
    ax.hist(df.loc[df["Class"] == 1, "Time"], bins=80, alpha=0.7, label="Fraud", color="#e76f51", density=True)
    ax.set_xlabel("Time (seconds from first transaction)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction time distribution by class")
    ax.legend()
    _savefig(fig, fig_dir / "03_time_by_class.png")

    # 4. Hourly fraud rate
    seconds_per_day = int(cfg["preprocessing"]["seconds_per_day"])
    hour = ((df["Time"] % seconds_per_day) / seconds_per_day * 24).astype(int)
    tmp = df.copy()
    tmp["hour"] = hour
    hourly = tmp.groupby("hour")["Class"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly.index, hourly["mean"], marker="o", color="#e76f51")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Hourly fraud rate")
    ax.set_xticks(range(0, 24, 2))
    _savefig(fig, fig_dir / "04_hourly_fraud_rate.png")
    summary["hourly_fraud_rate"] = hourly["mean"].to_dict()

    # 5. Correlation heatmap (sample for speed if large)
    sample = df if len(df) <= 80000 else df.sample(80000, random_state=cfg["project"]["seed"])
    corr = sample[feature_v + ["Amount", "Class"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, xticklabels=True, yticklabels=True, cbar_kws={"label": "Pearson r"})
    ax.set_title("Feature correlation heatmap (incl. Class)")
    _savefig(fig, fig_dir / "05_correlation_heatmap.png")

    # 6. Top discriminative features by |corr with Class|
    class_corr = corr["Class"].drop("Class").abs().sort_values(ascending=False)
    top = class_corr.head(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#264653")
    ax.set_xlabel("|Pearson correlation with Class|")
    ax.set_title("Top discriminative features (linear association)")
    _savefig(fig, fig_dir / "06_top_discriminative_features.png")
    summary["top_corr_with_class"] = class_corr.head(15).to_dict()

    # 7. Boxplot Amount by class
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_df = df.copy()
    plot_df["ClassLabel"] = plot_df["Class"].map({0: "Genuine", 1: "Fraud"})
    sns.boxplot(
        data=plot_df,
        x="ClassLabel",
        y="Amount",
        hue="ClassLabel",
        ax=ax,
        showfliers=False,
        palette={"Genuine": "#2a9d8f", "Fraud": "#e76f51"},
        legend=False,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Amount")
    ax.set_title("Amount by class (outliers hidden)")
    _savefig(fig, fig_dir / "07_amount_boxplot.png")

    out_json = resolve_path(cfg, "figures_dir").parent / "eda_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[eda] Summary → {out_json.relative_to(ROOT)}")
    _write_notebook_stub()
    return summary


def _write_notebook_stub() -> None:
    """Create / refresh a simple Jupyter notebook that calls the EDA module."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 01 — EDA: Credit Card Fraud (ULB)\n",
                    "\n",
                    "Run the project EDA module to regenerate figures under `reports/figures/`.\n",
                    "\n",
                    "**Why accuracy is misleading:** with ~0.17% fraud, a model that always predicts "
                    "genuine scores 99.83% accuracy while catching zero fraud.",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(ROOT))\n",
                    "from src.data.eda import run_eda\n",
                    "summary = run_eda()\n",
                    "summary['fraud_rate'], summary['n_fraud']",
                ],
            },
        ],
    }
    path = ROOT / "notebooks" / "01_eda.ipynb"
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[eda] Notebook stub → {path.relative_to(ROOT)}")


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
