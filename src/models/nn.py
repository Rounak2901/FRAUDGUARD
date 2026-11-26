"""Deterministic feed-forward fraud MLP (PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FraudMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.25,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss for extreme class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


@dataclass
class TrainResult:
    model: FraudMLP
    history: dict
    best_val_pr_auc: float


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (128, 64, 32),
    dropout: float = 0.25,
    epochs: int = 40,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
    loss_type: str = "bce_weighted",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    device: str | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = FraudMLP(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    if loss_type == "focal":
        criterion: nn.Module = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    best_state = None
    best_pr = -1.0
    wait = 0
    history = {"train_loss": [], "val_pr_auc": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(Xv)).cpu().numpy()
        pr = _pr_auc(y_val, val_prob)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_pr_auc"].append(pr)
        sched.step(pr)
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model.cpu(), history=history, best_val_pr_auc=best_pr)


@torch.no_grad()
def predict_proba(model: FraudMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    return torch.sigmoid(model(xt)).cpu().numpy()
