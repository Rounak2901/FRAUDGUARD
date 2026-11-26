# Final Report Outline — Fraud Guard

## 1. Introduction
Problem: rare, costly online payment fraud; accuracy is misleading under ~0.17% base rate.

## 2. Objectives
PR-AUC under imbalance; uncertainty-aware review; Bayesian entity risk; cost-sensitive decisions; SHAP explanations; real-time prototype.

## 3. Literature / related work
Supervised fraud ML, cost-sensitive learning, Bayesian NNs / MC Dropout, Beta-Bernoulli conjugate updating, anomaly detection, SHAP.

## 4. Dataset & methodology
- ULB primary; PaySim for entities
- Time-based 70/15/15 split (no future leakage)
- Robust scaling fit on train only; causal entity features

## 5. System design
Architecture diagram (`docs/architecture.md`). Modules: baselines, MLP, MC Dropout, autoencoder, ensemble, cost engine, API, dashboard.

## 6. Experiments
- Baselines vs NN vs BNN vs ensemble
- Imbalance: class weights vs focal loss
- Calibration ECE before/after isotonic
- Uncertainty quality (error rate by uncertainty bin)
- Ablation table
- Cost at 0.5 vs optimised threshold
- Latency p50/p95

## 7. Results
Populate from `reports/results.json` and `reports/figures/`.

## 8. Dashboard & demo
Live monitor, review queue, entity Beta belief, cost simulator, Try It form.

## 9. Limitations
PCA anonymisation; synthetic IDs if used; simulated stream; not PCI-DSS; public data ≠ bank traffic.

## 10. Future work
GNNs, adversarial robustness, federated learning, drift detectors (ADWIN), Kafka/Redis, MLflow MLOps.

## 11. Conclusion
End-to-end, reproducible, cost-aware, uncertainty-aware fraud detection prototype on CPU.
