"""Upload da folha de referência e mapa track -> personagem."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.schemas import CharacterOut, TrackMapIn
from db.models import Character, Job, JobState
from db.session import get_session

router = APIRouter()

ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".flac", ".m4a"}


def _out(c: Character) -> CharacterOut:
    return CharacterOut(id=c.id, job_id=c.job_id, name=c.name, sheet_url=c.sheet_url,
                        voice_mode=c.voice_mode, track_ids=c.track_ids or [],
                        avatar_ready=bool(c.avatar_key))


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Sobe uma folha (imagem) ou amostra de voz e devolve a URL s3:// para usar no JobIn."""
    from storage import s3
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(422, f"extensão não aceita: {ext or '(nenhuma)'}")
    key = f"uploads/{uuid.uuid4()}{ext}"
    s3.client().upload_fileobj(file.file, s3.bucket(), key,
                               ExtraArgs={"ContentType": file.content_type or "application/octet-stream"})
    return {"url": f"s3://{s3.bucket()}/{key}"}


@router.get("", response_model=list[CharacterOut])
def list_characters(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job não encontrado")
    return [_out(c) for c in job.characters]


@router.get("/{character_id}", response_model=CharacterOut)
def get_character(character_id: str, session: Session = Depends(get_session)):
    c = session.get(Character, character_id)
    if not c:
        raise HTTPException(404, "personagem não encontrado")
    return _out(c)


@router.put("/{character_id}/tracks", response_model=CharacterOut)
def map_tracks(character_id: str, payload: TrackMapIn, session: Session = Depends(get_session)):
    """Define à mão quais tracks (IDs do ByteTrack) viram este personagem. Um
    track só pode pertencer a um personagem; ele é removido dos outros. Vale
    para blocos ainda não renderizados (renderizados continuam como estão)."""
    c = session.get(Character, character_id)
    if not c:
        raise HTTPException(404, "personagem não encontrado")
    if c.job.state in (JobState.done, JobState.cancelled):
        raise HTTPException(409, f"job está {c.job.state.value}")
    ids = sorted(set(payload.track_ids))
    for other in c.job.characters:
        if other.id != c.id and set(other.track_ids or []) & set(ids):
            other.track_ids = [t for t in other.track_ids if t not in ids]
    c.track_ids, c.track_ids_manual = ids, True
    return _out(c)
