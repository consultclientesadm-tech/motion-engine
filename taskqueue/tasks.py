"""Enfileira e executa o pipeline. Cada etapa é idempotente: rodar `run_job`
de novo (retomada, worker que caiu, teto aumentado) continua de onde parou."""
from __future__ import annotations

import logging
import threading

import redis

from api.budget import BudgetExceeded
from db.models import JobState
from db.session import session_scope
from settings import get_settings
from taskqueue.celery_app import app
from workers import pipeline

log = logging.getLogger(__name__)

# Trava curta renovada por uma thread: se a máquina morrer, ela expira em
# LOCK_TTL segundos (antes era 30 min, e o job ficava preso em "queued").
LOCK_TTL = 120
LOCK_RENEW_EVERY = 30


def lock_key(job_id: str) -> str:
    return f"job-lock:{job_id}"


def job_lock(job_id: str):
    # thread_local=False: a trava é renovada pela thread de keep-alive
    return redis.Redis.from_url(get_settings().redis_url).lock(
        lock_key(job_id), timeout=LOCK_TTL, thread_local=False)


def is_locked(job_id: str) -> bool:
    """Há um worker vivo segurando este job?"""
    return bool(redis.Redis.from_url(get_settings().redis_url).exists(lock_key(job_id)))


def _keep_alive(lock, stop: threading.Event) -> None:
    while not stop.wait(LOCK_RENEW_EVERY):
        try:
            lock.reacquire()
        except redis.exceptions.LockError:
            log.warning("perdi a trava %s", lock.name)
            return


def enqueue_job(job_id: str) -> str:
    r = run_job.apply_async(args=[job_id], queue="gpu")
    with session_scope() as s:
        from db.models import Job
        job = s.get(Job, job_id)
        job.celery_task_id = r.id
    return r.id


@app.task(bind=True, name="taskqueue.tasks.run_job", max_retries=30)
def run_job(self, job_id: str) -> str:
    from db.models import Job
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None or job.state in (JobState.done, JobState.cancelled):
            return job.state.value if job else "missing"
    lock = job_lock(job_id)
    if not lock.acquire(blocking=False):
        # outro worker está nele, ou a trava de um worker morto ainda não expirou:
        # tenta de novo depois (antes devolvia "locked" e o job ficava parado)
        log.warning("job %s travado por outro worker; nova tentativa em %ss", job_id, LOCK_TTL // 2)
        raise self.retry(countdown=LOCK_TTL // 2)
    stop = threading.Event()
    threading.Thread(target=_keep_alive, args=(lock, stop), daemon=True).start()
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
        stop.set()
        ctx.cleanup()
        try:
            lock.release()
        except redis.exceptions.LockError:
            pass
