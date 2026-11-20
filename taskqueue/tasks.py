"""Enfileira e executa o pipeline. Cada etapa é idempotente: rodar `run_job`
de novo (retomada, worker que caiu, teto aumentado) continua de onde parou."""
from __future__ import annotations

import logging

import redis

from api.budget import BudgetExceeded
from db.models import JobState
from db.session import session_scope
from settings import get_settings
from taskqueue.celery_app import app
from workers import pipeline

log = logging.getLogger(__name__)


def enqueue_job(job_id: str) -> str:
    r = run_job.apply_async(args=[job_id], queue="gpu")
    with session_scope() as s:
        from db.models import Job
        job = s.get(Job, job_id)
        job.celery_task_id = r.id
    return r.id


@app.task(bind=True, name="taskqueue.tasks.run_job")
def run_job(self, job_id: str) -> str:
    lock = redis.Redis.from_url(get_settings().redis_url).lock(f"job-lock:{job_id}", timeout=1800)
    if not lock.acquire(blocking=False):
        log.warning("job %s já está rodando em outro worker", job_id)
        return "locked"
    ctx = pipeline.Ctx(job_id=job_id, lock=lock)
    try:
        pipeline.run(ctx)
        return "done"
    except pipeline.Cancelled:
        return "cancelled"
    except BudgetExceeded as e:
        pipeline.set_state(job_id, JobState.paused_budget, error=str(e))
        return "paused_budget"
    except Exception as e:  # noqa: BLE001 - registra e marca como falho
        log.exception("job %s falhou", job_id)
        pipeline.set_state(job_id, JobState.failed, error=f"{type(e).__name__}: {e}")
        return "failed"
    finally:
        ctx.cleanup()
        try:
            lock.release()
        except redis.exceptions.LockError:
            pass
