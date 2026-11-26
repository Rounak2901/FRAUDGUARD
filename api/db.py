"""SQLAlchemy models and session helpers (SQLite by default)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import load_config

Base = declarative_base()
_engine = None
SessionLocal = None


class TransactionRow(Base):
    __tablename__ = "transactions"
    id = Column(String(64), primary_key=True)
    timestamp = Column(Float)
    entity_id = Column(String(64), index=True)
    merchant_id = Column(String(64), index=True)
    amount = Column(Float)
    features_json = Column(Text)
    true_label = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionRow(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), index=True)
    p_fraud = Column(Float)
    uncertainty = Column(Float)
    anomaly_score = Column(Float)
    entity_risk = Column(Float)
    risk_score = Column(Float)
    decision = Column(String(16), index=True)
    shap_json = Column(Text)
    latency_ms = Column(Float)
    model_version = Column(String(32))


class EntityBeliefRow(Base):
    __tablename__ = "entity_beliefs"
    entity_id = Column(String(64), primary_key=True)
    alpha = Column(Float)
    beta = Column(Float)
    updated_at = Column(DateTime)


class EntityBeliefHistoryRow(Base):
    __tablename__ = "entity_belief_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), index=True)
    timestamp = Column(DateTime)
    alpha = Column(Float)
    beta = Column(Float)


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), index=True)
    analyst_label = Column(Integer)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db(db_url: str | None = None):
    global _engine, SessionLocal
    cfg = load_config()
    url = db_url or cfg["api"]["db_url"]
    _engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=_engine)
    return _engine


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
