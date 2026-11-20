"""Teto de gasto por job e por dia. É o que te protege da fatura.

Diferença para o esqueleto original: em vez de cobrar *depois* de rodar e só
checar `gasto_hoje < teto`, aqui cada etapa/bloco **reserva** o custo previsto
*antes* de rodar. A reserva trava a linha do job e a do dia (SELECT ... FOR
UPDATE), então dois workers em paralelo nunca furam o teto juntos. Depois da
etapa, `settle` acerta previsto x medido (GPU-segundos reais).

  * `check_budget`  -> API, na criação (estimativa cabe no teto do job e no saldo do dia?)
  * `reserve`       -> worker, antes de cada etapa/bloco; levanta BudgetExceeded
  * `settle`        -> worker, depois; nunca bloqueia (o trabalho já foi feito)
  * `charge`        -> atalho: cobra GPU-segundos medidos sem reserva prévia
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from api.costs import gpu_usd
from db.models import DailySpend, Job, Spend, utctoday
from settings import Settings, get_settings

EPS = 1e-9


class BudgetExceeded(Exception):
    def __init__(self, scope: str, needed: float, available: float):
        self.scope, self.needed, self.available = scope, needed, available
        super().__init__(f"teto {scope} atingido: precisa de US$ {needed:.2f}, "
                         f"sobra US$ {max(available, 0):.2f}")


def daily_cap(job_daily: float | None, s: Settings | None = None) -> float:
    """O teto diário do request nunca passa do teto do servidor (DAILY_BUDGET_USD)."""
    s = s or get_settings()
    return s.daily_budget_usd if job_daily is None else min(job_daily, s.daily_budget_usd)


def spent_today(session: Session, day: date | None = None) -> float:
    row = session.get(DailySpend, day or utctoday())
    return row.usd if row else 0.0


def spent_by_job(session: Session, job_id: str) -> float:
    return float(session.scalar(select(func.coalesce(func.sum(Spend.usd), 0.0))
                                .where(Spend.job_id == job_id)) or 0.0)


def check_budget(session: Session, estimated_usd: float, job_budget: float,
                 daily_budget: float | None = None) -> None:
    """Levanta BudgetExceeded se a estimativa não cabe. Usado na criação do job."""
    if job_budget <= 0:
        raise BudgetExceeded("do job", estimated_usd, 0.0)
    if estimated_usd > job_budget + EPS:
        raise BudgetExceeded("do job", estimated_usd, job_budget)
    left = daily_cap(daily_budget) - spent_today(session)
    if estimated_usd > left + EPS:
        raise BudgetExceeded("diário", estimated_usd, left)


def _lock_day(session: Session, day: date) -> DailySpend:
    ins = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
    session.execute(ins(DailySpend).values(day=day, usd=0.0).on_conflict_do_nothing(index_elements=["day"]))
    return session.execute(select(DailySpend).where(DailySpend.day == day).with_for_update()).scalar_one()


def _lock_job(session: Session, job_id: str) -> Job:
    return session.execute(select(Job).where(Job.id == job_id).with_for_update()).scalar_one()


def reserve(session: Session, job_id: str, stage: str, usd: float) -> float:
    """Reserva `usd` para `stage` ou levanta BudgetExceeded. Trava job -> dia (ordem fixa)."""
    day = utctoday()
    job = _lock_job(session, job_id)
    daily = _lock_day(session, day)
    job_left = job.budget_usd - job.cost_usd
    if usd > job_left + EPS:
        raise BudgetExceeded("do job", usd, job_left)
    day_left = daily_cap(job.daily_budget_usd) - daily.usd
    if usd > day_left + EPS:
        raise BudgetExceeded("diário", usd, day_left)
    _book(session, job, daily, stage, "reserve", usd)
    return usd


def settle(session: Session, job_id: str, stage: str, reserved_usd: float, actual_usd: float) -> float:
    delta = round(actual_usd - reserved_usd, 6)
    if abs(delta) < EPS:
        return 0.0
    job = _lock_job(session, job_id)
    daily = _lock_day(session, utctoday())
    _book(session, job, daily, stage, "settle", delta)
    return delta


def charge(session: Session, job_id: str, seconds: float, stage: str = "gpu") -> float:
    """Compatível com o esqueleto: cobra GPU-segundos medidos, sem checar teto."""
    usd = gpu_usd(seconds)
    job = _lock_job(session, job_id)
    daily = _lock_day(session, utctoday())
    _book(session, job, daily, stage, "settle", usd)
    return usd


def _book(session: Session, job: Job, daily: DailySpend, stage: str, kind: str, usd: float) -> None:
    job.cost_usd = max(0.0, round(job.cost_usd + usd, 6))
    daily.usd = max(0.0, round(daily.usd + usd, 6))
    session.add(Spend(job_id=job.id, day=daily.day, stage=stage, kind=kind, usd=usd))
    session.flush()
