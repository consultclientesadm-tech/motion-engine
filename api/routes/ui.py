"""Interface web de teste: uma página estática + upload de vídeo/imagem e
streaming do resultado. Sem login nem pagamento (fora do escopo por enquanto).

O streaming passa pela API porque a URL pré-assinada do S3 aponta para o
endpoint interno (ex.: http://minio:9000), que o navegador não alcança.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from db.models import Job, JobState
from db.session import get_session

router = APIRouter()

STATIC = Path(__file__).resolve().parents[1] / "static"
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO = {".wav", ".mp3", ".flac", ".m4a"}
CHUNK = 1 << 20


@router.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@router.get("/ui", include_in_schema=False)
def page():
    return FileResponse(STATIC / "index.html", media_type="text/html")


def _probe_seconds(path: Path) -> float | None:
    try:
        from workers.media import probe
        return round(probe(path).duration_s, 3) or None
    except Exception:  # noqa: BLE001 - sem ffprobe ou arquivo estranho: a API mede depois
        return None


@router.post("/ui/upload")
def upload(file: UploadFile = File(...)):
    """Sobe vídeo, imagem ou áudio para o S3. Para vídeo, devolve também a
    duração (medida aqui, no arquivo local, para o POST /jobs não precisar
    alcançar o S3 com ffprobe)."""
    from storage import s3
    ext = Path(file.filename or "").suffix.lower()
    kind = "video" if ext in VIDEO else "image" if ext in IMAGE else "audio" if ext in AUDIO else None
    if kind is None:
        raise HTTPException(422, f"extensão não aceita: {ext or '(nenhuma)'}")
    key = f"uploads/{uuid.uuid4()}{ext}"
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / f"in{ext}"
        with open(local, "wb") as f:
            shutil.copyfileobj(file.file, f, length=CHUNK)
        duration = _probe_seconds(local) if kind == "video" else None
        if kind == "video" and duration is None:
            raise HTTPException(422, "não consegui ler esse vídeo (o arquivo tem trilha de vídeo?)")
        s3.upload(local, key, file.content_type or "application/octet-stream")
    return {"url": f"s3://{s3.bucket()}/{key}", "kind": kind, "duration_s": duration,
            "filename": file.filename}


_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


@router.get("/ui/jobs/{job_id}/video", include_in_schema=False)
def video(job_id: str, download: bool = False, range: str | None = Header(None),
          session: Session = Depends(get_session)):
    """Toca (com Range, para o player poder pular) ou baixa o output.mp4."""
    from storage import s3
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job não encontrado")
    if job.state != JobState.done or not job.output_key:
        raise HTTPException(409, "o vídeo ainda não está pronto")

    params = {"Bucket": s3.bucket(), "Key": job.output_key}
    status = 200
    if range and _RANGE.match(range.strip()):
        params["Range"] = range.strip()
        status = 206
    obj = s3.client().get_object(**params)
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(obj["ContentLength"])}
    if status == 206 and obj.get("ContentRange"):
        headers["Content-Range"] = obj["ContentRange"]
    else:
        status = 200
    if download:
        headers["Content-Disposition"] = f'attachment; filename="motion-{job_id[:8]}.mp4"'
    body = obj["Body"]
    return StreamingResponse(body.iter_chunks(CHUNK), status_code=status, media_type="video/mp4",
                             headers=headers)
