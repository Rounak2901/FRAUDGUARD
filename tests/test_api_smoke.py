"""API smoke tests (require trained web_bundle)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from pathlib import Path
    from src.config import resolve_path, load_config

    cfg = load_config()
    bundle = resolve_path(cfg, "models_dir") / "web_bundle.joblib"
    if not bundle.exists():
        pytest.skip("web_bundle.joblib missing — run py -m src.train_web_bundle")
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["bundle_loaded"] is True


def test_score(client):
    r = client.post("/api/v1/score", json={"amount": 100, "time": 1000, "entity_id": "card_t", "features": {"V14": -2}})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert "top_reasons" in body
