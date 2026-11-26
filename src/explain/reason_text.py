"""Turn feature impacts into plain-English analyst reasons."""

from __future__ import annotations

from typing import Any


FRIENDLY = {
    "Amount_scaled": "Scaled transaction amount",
    "Amount": "Transaction amount",
    "amount_zscore": "Amount vs this card's usual spend",
    "hour_sin": "Time-of-day pattern (sin)",
    "hour_cos": "Time-of-day pattern (cos)",
    "tx_count_1h": "Transactions in the last hour",
    "tx_count_24h": "Transactions in the last 24 hours",
    "entity_risk": "Bayesian entity fraud belief",
    "anomaly_score": "Unsupervised anomaly score",
}


def friendly_name(feature: str) -> str:
    if feature in FRIENDLY:
        return FRIENDLY[feature]
    if feature.startswith("V") and feature[1:].isdigit():
        return f"Unusual pattern in PCA feature {feature}"
    return feature.replace("_", " ")


def impact_to_text(feature: str, impact: float, value: float | None = None) -> str:
    direction = "toward fraud" if impact > 0 else "toward genuine"
    name = friendly_name(feature)
    if feature == "amount_zscore" and value is not None and abs(value) >= 2:
        return f"Amount is {abs(value):.1f}x std from this card's usual spend ({direction})"
    if feature.startswith("V"):
        return f"{name} pushed {direction} (impact {impact:+.3f})"
    return f"{name} pushed {direction} (impact {impact:+.3f})"


def top_reasons(impacts: dict[str, float], values: dict[str, float] | None = None, k: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    values = values or {}
    out = []
    for feat, imp in ranked:
        out.append(
            {
                "feature": feat,
                "impact": float(imp),
                "text": impact_to_text(feat, float(imp), values.get(feat)),
            }
        )
    return out
