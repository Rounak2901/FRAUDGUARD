# Fraud Guard — Architecture

```
[Transaction Simulator / Replay]
            |
            v
[FastAPI Scoring Service]  <-->  [Model Registry (models/)]
   1. Validate (Pydantic)
   2. Feature builder (+ rolling entity features)
   3. Bayesian entity-risk lookup (Beta)
   4. NN / MC-Dropout inference
   5. Anomaly score (autoencoder / Isolation Forest)
   6. Cost-sensitive decision (Approve / Review / Block)
   7. SHAP / surrogate local reasons
            |
            v
[SQLite: transactions, predictions, entity beliefs, feedback]
            |
            v
[REST + WebSocket] --> [React Dashboard]
```

## Dataset roles

| Dataset | Module |
|---------|--------|
| ULB creditcard (or synthetic ULB-like) | Supervised NN/BNN, baselines, anomaly, SHAP, API scoring |
| PaySim sample (or synthetic) | Entity Bayesian trajectories |
| Synthetic card_id/merchant_id on ULB | Fallback when PaySim absent (seed=42) |

## Latency budget

Target p95 < 100 ms on CPU for `/api/v1/score` (web path uses calibrated RF + sampled tree uncertainty for speed; full MC-Dropout BNN used offline in `src.train`).
