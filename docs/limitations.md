# Limitations and Future Work

## Limitations
- ULB features are PCA-anonymised → semantic SHAP text is approximate.
- Entity IDs may be synthetic (fixed seed) when PaySim is unavailable.
- Stream is a replay simulator, not live bank traffic.
- Not production security / PCI-DSS compliant.
- Public-data metrics may not transfer to real banks.
- Web serving path uses a calibrated RF + tree uncertainty for CPU latency; full MC-Dropout BNN is trained offline (`py -m src.train`).

## Future work
- Graph neural networks on transaction networks
- Adversarial robustness
- Federated learning across institutions
- Online learning with drift detection (ADWIN, Page-Hinkley)
- Kafka + Redis for production streaming
- Full MLOps with MLflow
