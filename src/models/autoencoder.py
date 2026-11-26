"""Autoencoder anomaly detector trained on genuine transactions only."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FraudAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims: Sequence[int] = (32, 16, 8)) -> None:
        super().__init__()
        dims = [n_features, *hidden_dims]
        enc = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        dec = []
        rev = list(reversed(dims))
        for a, b in zip(rev[:-1], rev[1:]):
            dec += [nn.Linear(a, b)]
            if b != n_features:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_genuine: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (32, 16, 8),
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> FraudAutoencoder:
    torch.manual_seed(seed)
    model = FraudAutoencoder(X_genuine.shape[1], hidden_dims=hidden_dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_genuine, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            loss.backward()
            opt.step()
    model.eval()
    return model.cpu()


@torch.no_grad()
def reconstruction_error(model: FraudAutoencoder, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    recon = model(xt)
    err = ((recon - xt) ** 2).mean(dim=1).cpu().numpy()
    return err


def normalize_scores(scores: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    span = max(ref_max - ref_min, 1e-9)
    return np.clip((scores - ref_min) / span, 0.0, 1.0)
