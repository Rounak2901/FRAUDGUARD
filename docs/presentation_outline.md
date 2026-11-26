# Presentation Outline (12–15 slides)

1. **Title** — Fraud Guard: Real-Time Online Transaction Fraud Detection  
2. **Problem** — 0.17% fraud; why accuracy fails  
3. **Objectives** — uncertainty, Bayesian entity risk, cost, SHAP, live API  
4. **Architecture** — diagram  
5. **Data & split** — ULB + PaySim; time-based split  
6. **Models** — baselines → MLP → MC Dropout BNN → autoencoder  
7. **Bayesian entity risk** — Beta(α,β) update live  
8. **Cost-sensitive decisions** — Approve / Review / Block  
9. **Results** — PR-AUC table + ablation  
10. **Calibration & uncertainty** — ECE + error vs uncertainty  
11. **Latency** — p50/p95 on CPU  
12. **Live demo** — dashboard walkthrough  
13. **Limitations & future work**  
14. **Conclusion & Q&A**
