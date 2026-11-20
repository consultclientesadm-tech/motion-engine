import pytest

from api.budget import BudgetExceeded, check_budget, reserve, settle, spent_today
from db.models import Job


def make_job(s, budget=10.0, daily=90.0):
    job = Job(video_url="x.mp4", budget_usd=budget, daily_budget_usd=daily, duration_s=60)
    s.add(job)
    s.flush()
    return job.id


def test_reserve_until_job_cap(db):
    with db.session_scope() as s:
        jid = make_job(s, budget=1.0)
    with db.session_scope() as s:
        reserve(s, jid, "a", 0.6)
    with pytest.raises(BudgetExceeded) as e, db.session_scope() as s:
        reserve(s, jid, "b", 0.6)
    assert e.value.scope == "do job"
    with db.session_scope() as s:
        assert s.get(Job, jid).cost_usd == pytest.approx(0.6)
        assert spent_today(s) == pytest.approx(0.6)


def test_daily_cap_across_jobs(db):
    with db.session_scope() as s:
        a, b = make_job(s, budget=50, daily=1.0), make_job(s, budget=50, daily=1.0)
    with db.session_scope() as s:
        reserve(s, a, "x", 0.8)
    with pytest.raises(BudgetExceeded) as e, db.session_scope() as s:
        reserve(s, b, "x", 0.3)
    assert e.value.scope == "diário"


def test_request_daily_cannot_exceed_server_daily(db):
    with db.session_scope() as s:
        jid = make_job(s, budget=200, daily=10_000)  # servidor: 90
    with pytest.raises(BudgetExceeded), db.session_scope() as s:
        reserve(s, jid, "x", 95)


def test_settle_refunds_and_charges(db):
    with db.session_scope() as s:
        jid = make_job(s)
    with db.session_scope() as s:
        reserve(s, jid, "r", 1.0)
        settle(s, jid, "r", 1.0, 0.4)
    with db.session_scope() as s:
        assert s.get(Job, jid).cost_usd == pytest.approx(0.4)
        assert spent_today(s) == pytest.approx(0.4)


def test_check_budget(db):
    with db.session_scope() as s:
        check_budget(s, 20, 30, 90)
        with pytest.raises(BudgetExceeded):
            check_budget(s, 31, 30, 90)
        with pytest.raises(BudgetExceeded):
            check_budget(s, 20, 30, 10)
