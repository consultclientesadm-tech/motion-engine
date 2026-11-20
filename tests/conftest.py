from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("DAILY_BUDGET_USD", "90")
os.environ.setdefault("GPU_USD_PER_HOUR", "1.2")


@pytest.fixture()
def db():
    from db import session as dbs
    from db.models import Base
    eng = dbs.configure("sqlite://")
    Base.metadata.create_all(eng)
    yield dbs
    Base.metadata.drop_all(eng)


@pytest.fixture()
def client(db, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    from api.routes import jobs
    queued: list[str] = []
    monkeypatch.setattr(jobs, "enqueue_job", queued.append)
    monkeypatch.setattr(jobs, "revoke", lambda task_id: None)
    with TestClient(app) as c:
        c.queued = queued
        yield c
