"""Checkpoint e retomada por bloco.

Cada bloco guarda no Postgres a última etapa concluída (BlockStage) e a chave
S3 do resultado. A retomada:
  1. confere se o arquivo de cada etapa marcada como feita existe no S3 (se o
     objeto sumiu, por lifecycle ou falha no upload, o bloco volta uma etapa);
  2. pula os blocos que já passaram da etapa pedida.
Um manifesto JSON também é gravado no S3 para auditoria/recuperação manual.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from sqlalchemy.orm import Session

from db.models import Block, BlockStage, Job

ORDER = [BlockStage.pending, BlockStage.rendered, BlockStage.lipsynced, BlockStage.upscaled]
DONE_FOR_MUX = {BlockStage.upscaled, BlockStage.skipped}


def rank(stage: BlockStage) -> int:
    return len(ORDER) if stage == BlockStage.skipped else ORDER.index(stage)


def plan_blocks(duration_s: float, block_seconds: float) -> list[tuple[float, float]]:
    n = max(1, math.ceil(duration_s / block_seconds - 1e-9))
    return [(round(i * block_seconds, 3), round(min((i + 1) * block_seconds, duration_s), 3)) for i in range(n)]


def ensure_blocks(session: Session, job: Job, block_seconds: float) -> list[Block]:
    if not job.blocks:
        for i, (a, b) in enumerate(plan_blocks(job.duration_s, block_seconds)):
            job.blocks.append(Block(index=i, start_s=a, end_s=b))
        session.flush()
    return job.blocks


def pending(job: Job, stage: BlockStage) -> list[Block]:
    """Blocos que ainda não chegaram a `stage` (ignora pulados)."""
    return [b for b in job.blocks if b.stage != BlockStage.skipped and rank(b.stage) < rank(stage)]


def mark(block: Block, stage: BlockStage, key: str | None = None) -> None:
    block.stage = stage
    if stage == BlockStage.rendered:
        block.render_key = key
    elif stage == BlockStage.lipsynced:
        block.lipsync_key = key
    elif stage in (BlockStage.upscaled, BlockStage.skipped):
        block.final_key = key


def verify(job: Job, exists: Callable[[str | None], bool]) -> list[int]:
    """Rebaixa blocos cujo arquivo sumiu. Retorna os índices rebaixados."""
    fixed = []
    for b in job.blocks:
        before = b.stage
        if b.stage in (BlockStage.upscaled, BlockStage.skipped) and not exists(b.final_key):
            b.stage = BlockStage.lipsynced if b.stage == BlockStage.upscaled else BlockStage.pending
            b.final_key = None
        if b.stage == BlockStage.lipsynced and not exists(b.lipsync_key):
            b.stage, b.lipsync_key = BlockStage.rendered, None
        if b.stage == BlockStage.rendered and not exists(b.render_key):
            b.stage, b.render_key = BlockStage.pending, None
        if b.stage != before:
            fixed.append(b.index)
    return fixed


def blocks_done(job: Job) -> int:
    return sum(1 for b in job.blocks if b.stage in DONE_FOR_MUX)


def manifest(job: Job) -> dict:
    return {
        "job_id": job.id, "state": job.state.value, "cost_usd": job.cost_usd,
        "blocks": [{"i": b.index, "start_s": b.start_s, "end_s": b.end_s, "stage": b.stage.value,
                    "has_people": b.has_people, "render": b.render_key, "lipsync": b.lipsync_key,
                    "final": b.final_key} for b in job.blocks],
    }


# pesos do progresso: análise 10%, personagens 5%, blocos 80%, mux 5%
_BLOCK_W = {BlockStage.pending: 0.0, BlockStage.rendered: 0.6, BlockStage.lipsynced: 0.75,
            BlockStage.upscaled: 1.0, BlockStage.skipped: 1.0}


def progress(job: Job) -> float:
    from db.models import JobState
    if job.state == JobState.done:
        return 1.0
    if not job.blocks:
        return 0.0
    p = 0.10 * sum(1 for b in job.blocks if b.analysis_key) / len(job.blocks)
    if job.characters and all(c.avatar_key for c in job.characters):
        p += 0.05
    p += 0.80 * sum(_BLOCK_W[b.stage] for b in job.blocks) / len(job.blocks)
    return round(min(p, 0.99), 4)
