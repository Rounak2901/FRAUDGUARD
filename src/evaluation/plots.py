"""Save evaluation figures at 300 dpi."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


DPI = 300


def save_roc_pr(y_true, y_prob, out_dir: Path, prefix: str = "model") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC curve")
    fig.savefig(out_dir / f"{prefix}_roc.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision–Recall curve")
    fig.savefig(out_dir / f"{prefix}_pr.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_confusion(y_true, y_pred, out_dir: Path, prefix: str = "model") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_dir / f"{prefix}_confusion.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_reliability(y_true, y_prob, out_dir: Path, n_bins: int = 15, prefix: str = "model") -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    means, accs = [], []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if m.any():
            means.append(y_prob[m].mean())
            accs.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.plot(means, accs, "o-", color="#0ea5e9", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.savefig(out_dir / f"{prefix}_reliability.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_uncertainty_vs_error(y_true, y_prob, uncertainty, out_dir: Path, prefix: str = "bnn") -> None:
    err = (y_prob >= 0.5).astype(int) != y_true
    # Bin by uncertainty
    qs = np.quantile(uncertainty, [0, 0.33, 0.66, 1.0])
    rates, labels = [], []
    for i in range(3):
        m = (uncertainty >= qs[i]) & (uncertainty <= qs[i + 1])
        rates.append(float(err[m].mean()) if m.any() else 0.0)
        labels.append(f"Q{i+1}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, rates, color="#f43f5e")
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Uncertainty tertile")
    ax.set_title("Error rate vs uncertainty")
    fig.savefig(out_dir / f"{prefix}_uncertainty_error.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
