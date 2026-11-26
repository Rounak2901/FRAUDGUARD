# Fraud Guard — Project Output

**Repository:** https://github.com/Rounak2901/FRAUDGUARD  
**Version:** 1.0.0  
**Seed:** 42  
**Author:** Rounak Dhall  

This file is the consolidated **output / results deliverable** for the B.Tech CSE final-year project: *Fraud Detection System for Real-Time Online Transactions*.

---

## 1. What was delivered

| Deliverable | Location |
|-------------|----------|
| Source code (ML + API + dashboard) | `src/`, `api/`, `frontend/` |
| Config | `config.yaml` |
| Trained metrics JSON | `reports/results.json` |
| Figures (EDA + evaluation) | `reports/figures/` |
| Docs (architecture, report, slides, demo) | `docs/` |
| Unit / API tests (14 passed) | `tests/` |
| Docker | `Dockerfile`, `docker-compose.yml` |
| This output summary | `PROJECT_OUTPUT.md` |

---

## 2. How to reproduce outputs

```powershell
cd FRAUDGUARD   # or local clone path
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt

py -m src.data.download --synthetic
py -m src.data.eda
py -m src.data.preprocess
py -m src.train_web_bundle
py -m src.train
py -m src.evaluate
py -m pytest tests/ -v

# Web app
py -m uvicorn api.main:app --host 127.0.0.1 --port 8001
cd frontend; npm install; npm run dev
```

- API docs: http://127.0.0.1:8001/docs  
- Dashboard: http://127.0.0.1:5173 (or Vite’s printed port)

---

## 3. Dataset note (important)

Outputs below were produced on **synthetic ULB-like + PaySim-like data** (`--synthetic`) so the pipeline runs without Kaggle credentials.

For thesis-quality numbers, replace with real Kaggle ULB / PaySim CSVs and re-run `preprocess` + `train`.

| Dataset role | Module |
|--------------|--------|
| ULB (or synthetic ULB-like) | Classifier, BNN, anomaly, SHAP, API |
| PaySim sample (or synthetic) | Bayesian entity-risk |

Time-based split: **70% train / 15% val / 15% test** (no shuffle before cut).

---

## 4. Evaluation results (time-based test set)

Primary metric: **PR-AUC** (accuracy is not used as the main score).

### 4.1 Baselines

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 | MCC |
|-------|--------|---------|-----------|--------|----|----|-----|
| Logistic Regression | 0.9324 | 0.9996 | 0.4595 | 0.9444 | 0.6182 | 0.7798 | 0.6577 |
| Random Forest | 0.6303 | 0.9960 | 1.0000 | 0.1667 | 0.2857 | 0.2000 | 0.4078 |
| XGBoost | 0.8026 | 0.9981 | 0.7222 | 0.7222 | 0.7222 | 0.7222 | 0.7216 |
| Isolation Forest | 0.0565 | 0.9408 | 0.0139 | 0.8333 | 0.0273 | 0.0651 | 0.0963 |

### 4.2 Neural network

- Chosen loss (by val PR-AUC): **weighted BCE** (beat focal on this run)
- Val PR-AUC (BCE): **0.9804**
- Val PR-AUC (focal): **0.3150**
- **Test PR-AUC: 0.9265** | Test ROC-AUC: **0.9996**

### 4.3 Bayesian NN (MC Dropout, T=40) + calibration

| Metric | Value |
|--------|-------|
| Test PR-AUC (uncalibrated mean) | 0.9076 |
| Test PR-AUC (isotonic calibrated) | **0.8216** |
| ECE before → after | **0.1803 → 0.0005** |
| Mean uncertainty (correct) | higher errors should show higher σ (see figure) |

### 4.4 Ensemble (NN + entity Beta risk + anomaly)

| Item | Value |
|------|-------|
| Learned weights (val grid) | w_nn=0.20, w_entity=0.60, w_anomaly=0.20 |
| **Test PR-AUC** | **0.8908** |

### 4.5 Ablation (test)

| Component set | PR-AUC | Business cost @ 0.5 thr |
|---------------|--------|-------------------------|
| NN only | 0.8216 | 1096.9 |
| NN + entity | 0.8040 | 1096.9 |
| NN + anomaly | 0.8908 | 1078.0 |
| Full ensemble | 0.8908 | 5568.4* |

\*Cost at fixed 0.5 threshold can be high; **cost-optimal threshold** (below) is what the decision engine uses.

### 4.6 Cost-sensitive decision

| Setting | Total business cost (test) |
|---------|----------------------------|
| Threshold 0.5 | 5568.43 |
| **Val-optimised threshold** | **1013.24** |

Cost matrix (configurable in `config.yaml`): C_fp = 5, C_fn = amount × multiplier, plus Review band for high uncertainty.

---

## 5. Figures produced

Saved under `reports/figures/` at 300 dpi:

**EDA**
- `01_class_distribution.png`
- `02_amount_log_by_class.png`
- `03_time_by_class.png`
- `04_hourly_fraud_rate.png`
- `05_correlation_heatmap.png`
- `06_top_discriminative_features.png`
- `07_amount_boxplot.png`

**Model evaluation**
- `bnn_roc.png`, `bnn_pr.png`, `bnn_reliability.png`, `bnn_uncertainty_error.png`
- `ensemble_roc.png`, `ensemble_pr.png`, `ensemble_confusion.png`

Machine-readable dump: `reports/results.json`

---

## 6. Real-time system output

### API (`api/`)
- `POST /api/v1/score` — probability, uncertainty, anomaly, entity risk, decision, reasons, latency
- Batch score, transactions, feedback (Bayesian update), entity beliefs, live metrics
- Stream start/stop + WebSocket `/ws/live`
- Health: `/health`

### Dashboard (`frontend/`)
1. Live Monitor  
2. Transaction Detail  
3. Review Queue  
4. Entity Risk Explorer  
5. Model Insights  
6. Cost Simulator  
7. Try It Yourself  
8. About  

Typical scoring latency (web bundle, after warmup): ~**p50 ≈ 90–100 ms** on CPU.

---

## 7. Tests

```
14 passed
```

Coverage areas: time split, causal features, Beta entity update, cost engine, API health/score smoke.

---

## 8. Limitations (stated for honest reporting)

- Synthetic public-style data used for this output run (replace with Kaggle for final thesis tables).
- ULB features are PCA-anonymised → explanations are approximate.
- Entity IDs may be synthetic when PaySim is absent.
- Stream is a replay simulator, not a live bank feed.
- Not PCI-DSS / production security.

See `docs/limitations.md`.

---

## 9. Documentation pack

| File | Purpose |
|------|---------|
| `docs/architecture.md` | System diagram |
| `docs/report_outline.md` | Final report skeleton |
| `docs/presentation_outline.md` | 12–15 slides |
| `docs/demo_script.md` | 3-minute demo |
| `README.md` | Install & run |

---

## 10. Success criteria checklist

| Criterion | Status (this run) |
|-----------|-------------------|
| PR-AUC competitive with classical baselines | Yes (NN 0.93 ≈ LR 0.93; ensemble 0.89) |
| Lower cost at optimised threshold vs 0.5 | Yes (1013 vs 5568) |
| ECE improves after calibration | Yes (0.18 → 0.0005) |
| Working API + live dashboard | Yes |
| Reproducible train command | `py -m src.train` |
| One-command Docker path | `docker compose up --build` |

---

*Generated as the project output artifact for submission and demo. Re-run `py -m src.train` after swapping in real Kaggle data to refresh numbers in `reports/results.json` and this summary.*
