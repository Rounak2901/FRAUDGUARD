"""
Fraud Guard API — real-time scoring, feedback, simulator, WebSocket live feed.

Run: py -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.db import (
    EntityBeliefHistoryRow,
    EntityBeliefRow,
    FeedbackRow,
    PredictionRow,
    TransactionRow,
    get_db,
    init_db,
)
from api.schemas import (
    BatchScoreIn,
    CostSimIn,
    FeedbackIn,
    HealthOut,
    ScoreOut,
    StreamControl,
    TransactionIn,
)
from api.services.scoring import scoring_service
from api.simulator import simulator
from src.config import load_config
from src.decision.engine import DecisionConfig, decide

cfg = load_config()
app = FastAPI(title="Fraud Guard API", version=cfg["project"]["version"], docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"] + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory recent feed for REST fallback
RECENT: list[dict[str, Any]] = []
WS_CLIENTS: list[WebSocket] = []


def _persist(tx: TransactionIn, out: ScoreOut, values: dict) -> None:
    RECENT.insert(0, out.model_dump())
    del RECENT[500:]
    # DB persist best-effort
    try:
        db = next(get_db())
        db.merge(
            TransactionRow(
                id=out.transaction_id,
                timestamp=float(tx.time),
                entity_id=tx.entity_id,
                merchant_id=tx.merchant_id,
                amount=float(tx.amount),
                features_json=json.dumps(values),
                true_label=tx.true_label,
            )
        )
        db.add(
            PredictionRow(
                transaction_id=out.transaction_id,
                p_fraud=out.p_fraud,
                uncertainty=out.uncertainty_std,
                anomaly_score=out.anomaly_score,
                entity_risk=out.entity_risk_posterior,
                risk_score=out.risk_score,
                decision=out.decision,
                shap_json=json.dumps([r.model_dump() for r in out.top_reasons]),
                latency_ms=out.latency_ms,
                model_version=out.model_version,
            )
        )
        db.commit()
        db.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[db] persist skipped: {exc}")


@app.on_event("startup")
def startup() -> None:
    init_db()
    scoring_service.load()
    simulator.load_slice(cfg["simulator"]["dataset_slice"])

    async def broadcast(payload: dict) -> None:
        RECENT.insert(0, payload)
        del RECENT[500:]
        dead = []
        for ws in WS_CLIENTS:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            WS_CLIENTS.remove(ws)

    simulator.subscribe(broadcast)


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok" if scoring_service.ready else "degraded",
        model_version=scoring_service.version,
        bundle_loaded=scoring_service.ready,
    )


@app.post("/api/v1/score", response_model=ScoreOut)
def score_one(tx: TransactionIn) -> ScoreOut:
    try:
        return scoring_service.score(tx, persist_hooks={"save": _persist})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/score/batch")
def score_batch(body: BatchScoreIn) -> dict:
    results = [scoring_service.score(tx, persist_hooks={"save": _persist}) for tx in body.transactions]
    return {"results": [r.model_dump() for r in results], "n": len(results)}


@app.get("/api/v1/transactions")
def list_transactions(
    decision: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    rows = RECENT
    if decision:
        rows = [r for r in rows if r.get("decision") == decision.upper()]
    return {"total": len(rows), "items": rows[offset : offset + limit]}


@app.get("/api/v1/transactions/{tx_id}")
def get_transaction(tx_id: str):
    for r in RECENT:
        if r.get("transaction_id") == tx_id:
            return r
    raise HTTPException(status_code=404, detail="Transaction not found in recent memory")


@app.post("/api/v1/feedback")
def feedback(body: FeedbackIn, db: Session = Depends(get_db)):
    # Find entity from recent
    entity_id = "card_demo"
    for r in RECENT:
        if r.get("transaction_id") == body.transaction_id:
            entity_id = r.get("entity_id") or entity_id
            break
    belief = scoring_service.apply_feedback(entity_id, is_fraud=bool(body.analyst_label))
    db.add(FeedbackRow(transaction_id=body.transaction_id, analyst_label=int(body.analyst_label)))
    db.merge(
        EntityBeliefRow(
            entity_id=entity_id,
            alpha=belief["alpha"],
            beta=belief["beta"],
            updated_at=datetime.now(timezone.utc),
        )
    )
    db.add(
        EntityBeliefHistoryRow(
            entity_id=entity_id,
            timestamp=datetime.now(timezone.utc),
            alpha=belief["alpha"],
            beta=belief["beta"],
        )
    )
    db.commit()
    return {"ok": True, "entity_id": entity_id, "belief": belief}


@app.get("/api/v1/entities/{entity_id}")
def entity_belief(entity_id: str):
    peek = scoring_service.entity_store.peek(entity_id)
    hist = scoring_service.entity_store.history(entity_id)
    return {**peek, "history": hist}


@app.get("/api/v1/metrics")
def live_metrics():
    m = scoring_service.metrics_live
    scored = max(m["scored"], 1)
    return {
        "volume": m["scored"],
        "fraud_labeled": m["fraud_labeled"],
        "block_rate": m["blocked"] / scored,
        "review_rate": m["reviewed"] / scored,
        "approve_rate": m["approved"] / scored,
        "avg_latency_ms": m["latency_sum_ms"] / scored,
        "amount_blocked": m["amount_blocked"],
        "cost_saved_est": m["cost_saved_est"],
        "simulator_running": simulator.running,
        "simulator_rate_tps": simulator.rate_tps,
        "model_metrics": (scoring_service.bundle or {}).get("metrics", {}),
    }


@app.post("/api/v1/stream/start")
async def stream_start(body: StreamControl):
    return await simulator.start(rate_tps=body.rate_tps, dataset_slice=body.dataset_slice)


@app.post("/api/v1/stream/stop")
async def stream_stop():
    return await simulator.stop()


@app.post("/api/v1/cost/simulate")
def cost_simulate(body: CostSimIn):
    """Recompute decisions on recent feed under new cost knobs (dashboard Cost Simulator)."""
    cfg_dec = DecisionConfig(
        c_fp=body.c_fp,
        c_fn_multiplier=body.c_fn_multiplier,
        c_review=body.c_review,
        review_uncertainty_threshold=scoring_service.decision_cfg.review_uncertainty_threshold,
        review_margin=scoring_service.decision_cfg.review_margin,
        block_threshold=scoring_service.decision_cfg.block_threshold,
    )
    # Also update live engine
    scoring_service.decision_cfg = cfg_dec
    totals = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0, "total_expected_cost": 0.0}
    curve = []
    for thr in [i / 20 for i in range(1, 20)]:
        cost = 0.0
        for r in RECENT[:200]:
            p = float(r.get("risk_score", r.get("p_fraud", 0)))
            amount = float(r.get("amount") or 100)
            if p >= thr:
                cost += (1 - p) * body.c_fp
            else:
                cost += p * amount * body.c_fn_multiplier
        curve.append({"threshold": thr, "total_cost": cost})
    for r in RECENT[:200]:
        d = decide(
            p_fraud=float(r.get("p_fraud", 0)),
            amount=float(r.get("amount") or 100),
            uncertainty=float(r.get("uncertainty_std", 0)),
            anomaly=float(r.get("anomaly_score", 0)),
            entity_risk=float(r.get("entity_risk_posterior", 0)),
            cfg=cfg_dec,
            weights=scoring_service.weights,  # type: ignore[arg-type]
        )
        totals[d.decision] += 1
        totals["total_expected_cost"] += (
            d.expected_loss_if_approved if d.decision == "APPROVE" else cfg_dec.c_fp if d.decision == "BLOCK" else cfg_dec.c_review
        )
    best = min(curve, key=lambda c: c["total_cost"]) if curve else {"threshold": 0.5, "total_cost": 0}
    return {"totals": totals, "cost_curve": curve, "optimal_threshold": best}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    WS_CLIENTS.append(websocket)
    try:
        for item in reversed(RECENT[:30]):
            await websocket.send_json(item)
        while True:
            # Keep alive; client may send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in WS_CLIENTS:
            WS_CLIENTS.remove(websocket)
