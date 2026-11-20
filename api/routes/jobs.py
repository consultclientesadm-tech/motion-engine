from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import costs
from api.budget import BudgetExceeded, check_budget
from api.schemas import EstimateIn, JobIn, JobStatus, ResumeIn
from db.models import ACTIVE_STATES, Character, Job, JobState
from db.session import get_session
from settings import get_settings
from workers.render import checkpoint

router = APIRouter()


def enqueue_job(job_id: str) -> None:
    from taskqueue.tasks import enqueue_job as _enqueue  # import tardio: testes trocam esta função
    _enqueue(job_id)


def revoke(task_id: str | None) -> None:
    if task_id:
        from taskqueue.celery_app import app
        app.control.revoke(task_id)  # sem terminate: o worker para sozinho no próximo bloco


def to_status(job: Job) -> JobStatus:
    output_url = None
    if job.output_key and job.state == JobState.done:
        from storage import s3
        output_url = s3.presigned_url(job.output_key)
    return JobStatus(
        id=job.id, state=job.state.value, progress=checkpoint.progress(job),
        cost_usd=round(job.cost_usd, 4), blocks_done=checkpoint.blocks_done(job),
        blocks_total=len(job.blocks), output_url=output_url,
        budget_usd=job.budget_usd, estimated_usd=round(job.estimated_usd, 4), error=job.error,
    )


def probe_duration(url: str) -> float:
    from storage import s3
    from workers.media import probe
    target = url
    if url.startswith("s3://"):
        b, _, key = url[5:].partition("/")
        target = s3.client().generate_presigned_url("get_object", Params={"Bucket": b, "Key": key})
    return probe(target).duration_s


def _get(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job não encontrado")
    return job


@router.post("/estimate")
def estimate(payload: EstimateIn):
    est = costs.estimate(payload.duration_s, people=payload.people,
                         target_resolution=payload.target_resolution,
                         only_people=payload.process_only_with_people,
                         people_ratio=payload.people_ratio, speech_s=payload.speech_s,
                         lipsync=payload.speech_s > 0)
    return est.to_dict()


@router.post("", response_model=JobStatus, status_code=201)
def create_job(payload: JobIn, session: Session = Depends(get_session)):
    s = get_settings()
    if payload.budget_usd > s.max_job_budget_usd:
        raise HTTPException(422, f"budget_usd acima do máximo do servidor (US$ {s.max_job_budget_usd})")
    names = {c.name for c in payload.characters}
    if payload.script and any(ln.character not in names for ln in payload.script):
        raise HTTPException(422, "script cita personagem que não está em characters")

    duration = payload.duration_s
    if duration is None:
        try:
            duration = probe_duration(payload.video_url)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"não consegui medir o vídeo ({e}); envie duration_s") from e

    speech = sum(len(ln.text) for ln in payload.script or []) / 15.0
    est = costs.estimate(duration, people=len(payload.characters),
                         target_resolution=payload.target_resolution,
                         only_people=payload.process_only_with_people,
                         speech_s=speech, lipsync=bool(payload.script))
    try:
        check_budget(session, est.total_usd, payload.budget_usd, payload.daily_budget_usd)
    except BudgetExceeded as e:
        raise HTTPException(402, {"message": str(e), "scope": e.scope,
                                  "estimated_usd": est.total_usd, "available_usd": round(e.available, 4)}) from e

    job = Job(video_url=payload.video_url, budget_usd=payload.budget_usd,
              daily_budget_usd=payload.daily_budget_usd, target_resolution=payload.target_resolution,
              process_only_with_people=payload.process_only_with_people, duration_s=duration,
              estimated_usd=est.total_usd,
              script=[ln.model_dump() for ln in payload.script] if payload.script else None)
    for i, c in enumerate(payload.characters):
        job.characters.append(Character(position=i, name=c.name, sheet_url=c.sheet_url,
                                        voice_ref_url=c.voice_ref_url, voice_mode=c.voice_mode))
    session.add(job)
    session.commit()  # o worker precisa enxergar o job antes de ser enfileirado
    enqueue_job(job.id)
    session.refresh(job)
    return to_status(job)


@router.get("", response_model=list[JobStatus])
def list_jobs(limit: int = 20, session: Session = Depends(get_session)):
    jobs = session.scalars(select(Job).order_by(Job.created_at.desc()).limit(min(limit, 100))).all()
    return [to_status(j) for j in jobs]


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str, session: Session = Depends(get_session)):
    return to_status(_get(session, job_id))


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, session: Session = Depends(get_session)):
    job = _get(session, job_id)
    if job.state in (JobState.done, JobState.cancelled):
        return {"ok": True, "state": job.state.value}
    job.state = JobState.cancelled
    session.commit()
    revoke(job.celery_task_id)
    return {"ok": True, "state": "cancelled"}


@router.post("/{job_id}/resume", response_model=JobStatus)
def resume_job(job_id: str, payload: ResumeIn | None = None, session: Session = Depends(get_session)):
    """Retoma do último bloco salvo. Use após paused_budget (com teto maior), failed ou cancelled."""
    job = _get(session, job_id)
    if job.state in ACTIVE_STATES:
        raise HTTPException(409, f"job já está em andamento ({job.state.value})")
    if job.state == JobState.done:
        raise HTTPException(409, "job já terminou")
    if payload and payload.budget_usd:
        if payload.budget_usd > get_settings().max_job_budget_usd:
            raise HTTPException(422, "budget_usd acima do máximo do servidor")
        job.budget_usd = payload.budget_usd
    if job.cost_usd >= job.budget_usd:
        raise HTTPException(402, f"o job já gastou US$ {job.cost_usd:.2f} do teto de US$ {job.budget_usd:.2f}; "
                                 "envie um budget_usd maior")
    job.state, job.error = JobState.queued, None
    session.commit()
    enqueue_job(job.id)
    session.refresh(job)
    return to_status(job)
