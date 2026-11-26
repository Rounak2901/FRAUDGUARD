"""Bayesian Neural Network via Monte Carlo Dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.nn import FraudMLP


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep Dropout layers in train mode while rest stays eval (BN frozen)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: FraudMLP,
    X: np.ndarray,
    n_passes: int = 40,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    T stochastic forward passes with dropout on.
    Returns mean probability, std (epistemic), predictive entropy, 95% CI.
    """
    model.to(device)
    enable_mc_dropout(model)
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    probs = []
    for _ in range(n_passes):
        logits = model(xt)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    stack = np.stack(probs, axis=0)  # (T, N)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    # Binary predictive entropy of the mean predictive distribution
    eps = 1e-9
    entropy = -(mean * np.log(mean + eps) + (1 - mean) * np.log(1 - mean + eps))
    lo = np.percentile(stack, 2.5, axis=0)
    hi = np.percentile(stack, 97.5, axis=0)
    return {
        "p_fraud_mean": mean,
        "uncertainty_std": std,
        "predictive_entropy": entropy,
        "ci_lo": lo,
        "ci_hi": hi,
        "samples": stack,
    }


class BayesLinear(nn.Module):
    """Minimal Bayes-by-Backprop linear layer (Gaussian variational weights)."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0) -> None:
        super().__init__()
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        b_sigma = torch.log1p(torch.exp(self.bias_rho))
        w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
        return x @ w.t() + b

    def kl(self) -> torch.Tensor:
        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        b_sigma = torch.log1p(torch.exp(self.bias_rho))
        prior = self.prior_sigma

        def _kl(mu, sigma):
            return (sigma.log() - np.log(prior) + (prior**2 + mu.pow(2)) / (2 * sigma.pow(2)) - 0.5).sum()

        # Use torch log for prior constant
        def kl_gauss(mu, sigma):
            return (torch.log(sigma) - np.log(prior) + (prior**2 + mu.pow(2)) / (2 * sigma.pow(2)) - 0.5).sum()

        return kl_gauss(self.weight_mu, w_sigma) + kl_gauss(self.bias_mu, b_sigma)


class BayesMLP(nn.Module):
    """Small BBB network for comparison with MC Dropout."""

    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.l1 = BayesLinear(n_features, hidden)
        self.l2 = BayesLinear(hidden, 32)
        self.l3 = BayesLinear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return self.l3(x).squeeze(-1)

    def kl(self) -> torch.Tensor:
        return self.l1.kl() + self.l2.kl() + self.l3.kl()
