"""Celery. O pacote se chama `taskqueue` (e não `queue`, como no esboço) porque
`queue` é um módulo da biblioteca padrão do Python que o próprio Celery/kombu
importa; uma pasta `queue/` na raiz quebraria o worker."""
from __future__ import annotations

from celery import Celery

from settings import get_settings

s = get_settings()
app = Celery("motion_engine", broker=s.redis_url, backend=s.redis_url, include=["taskqueue.tasks"])
app.conf.update(
    task_acks_late=True,                 # worker caiu -> tarefa volta pra fila e retoma do checkpoint
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,        # 1 job por GPU de cada vez
    broker_transport_options={"visibility_timeout": 24 * 3600},  # jobs longos não são reentregues
    task_default_queue="gpu",
    task_routes={"taskqueue.tasks.run_job": {"queue": "gpu"}},
    task_track_started=True,
    result_expires=7 * 24 * 3600,
    timezone="UTC",
)
