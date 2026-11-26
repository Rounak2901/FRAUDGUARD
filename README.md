# Fraud Guard

**Real-time online transaction fraud detection** — B.Tech CSE final-year project (AI/ML).

Neural network + MC-Dropout uncertainty · Bayesian entity-risk (Beta) · anomaly detection · cost-sensitive decisions · SHAP / surrogate explanations · FastAPI scoring · live React dashboard.

---

## Quick start (local)

```powershell
cd "C:\Users\Rounak\Desktop\FRAUD GUARD"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt

py -m src.data.download --synthetic   # or omit --synthetic with Kaggle API
py -m src.data.eda
py -m src.data.preprocess
py -m src.train_web_bundle            # fast path for the web app
py -m src.train                       # full academic pipeline (NN/BNN/AE/ablation)

# API
py -m uvicorn api.main:app --host 127.0.0.1 --port 8001

# Dashboard (new terminal)
cd frontend
npm install
npm run dev
```

- Dashboard: http://127.0.0.1:5173 (or the port Vite prints)  
- API docs: http://127.0.0.1:8001/docs  

### Docker

```powershell
docker compose up --build
```

- UI: http://localhost:5173  
- API: http://localhost:8001/docs  

---

## Dataset roles

| Dataset | Feeds |
|---------|--------|
| **ULB Credit Card Fraud** | Supervised models, BNN, anomaly, SHAP, cost engine, API |
| **PaySim sample** | Bayesian entity-risk trajectories |
| Fallback | Synthetic ULB-like + PaySim-like data (`--synthetic`) |

---

## Commands

| Command | Purpose |
|---------|---------|
| `py -m src.data.download` | Fetch / synthesise data |
| `py -m src.data.eda` | EDA figures → `reports/figures/` |
| `py -m src.data.preprocess` | Time split + scaler |
| `py -m src.models.baselines` | Classical baselines only |
| `py -m src.train` | Full train + `reports/results.json` |
| `py -m src.evaluate` | Print metrics summary |
| `py -m src.train_web_bundle` | Fast sklearn bundle for API |
| `py -m pytest tests/ -v` | Unit + API smoke tests |

---

## Project layout

See `docs/architecture.md`. Key folders: `src/`, `api/`, `frontend/`, `notebooks/`, `tests/`, `docs/`, `models/`, `reports/`.

## Documentation

- `docs/architecture.md` — system diagram  
- `docs/report_outline.md` — thesis report skeleton  
- `docs/presentation_outline.md` — 12–15 slides  
- `docs/demo_script.md` — 3-minute demo  
- `docs/limitations.md` — limitations & future work  

## License

Academic / educational use. Dataset licenses remain with their publishers.
