"""Contrato de dados da API. CharacterIn, JobIn e JobStatus seguem o contrato
original; os demais são acréscimos (marcados)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

State = Literal["queued", "analyzing", "building", "rendering", "audio", "post",
                "done", "paused_budget", "failed", "cancelled"]  # + cancelled


class CharacterIn(BaseModel):
    name: str
    sheet_url: str                       # folha de referência
    voice_ref_url: str | None = None     # amostra pra clonar
    voice_mode: Literal["clone", "generate"] = "clone"


class Line(BaseModel):
    """Acréscimo: fala opcional. Sem `script`, o job mantém o áudio original."""
    character: str                       # nome do personagem
    start_s: float = Field(ge=0)
    text: str


class JobIn(BaseModel):
    video_url: str
    characters: list[CharacterIn] = Field(min_length=1, max_length=8)
    budget_usd: float = Field(30.0, gt=0)          # teto por job
    daily_budget_usd: float = Field(90.0, gt=0)    # teto por dia (limitado ao do servidor)
    target_resolution: Literal["720p", "1080p"] = "1080p"
    process_only_with_people: bool = True
    # acréscimos
    duration_s: float | None = Field(None, gt=0, description="se omitido, a API mede com ffprobe")
    script: list[Line] | None = None

    @field_validator("characters")
    @classmethod
    def unique_names(cls, v: list[CharacterIn]):
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError("nomes de personagem devem ser únicos")
        return v


class JobStatus(BaseModel):
    id: str
    state: State
    progress: float
    cost_usd: float
    blocks_done: int
    blocks_total: int
    output_url: str | None = None
    # acréscimos
    budget_usd: float | None = None
    estimated_usd: float | None = None
    error: str | None = None


class CharacterOut(BaseModel):
    id: str
    job_id: str
    name: str
    sheet_url: str
    voice_mode: str
    track_ids: list[int]
    avatar_ready: bool


class TrackMapIn(BaseModel):
    track_ids: list[int]


class ResumeIn(BaseModel):
    budget_usd: float | None = Field(None, gt=0, description="novo teto do job, se parou por orçamento")


class EstimateIn(BaseModel):
    duration_s: float = Field(gt=0)
    people: int = Field(3, ge=1, le=8)
    target_resolution: Literal["720p", "1080p"] = "1080p"
    process_only_with_people: bool = True
    people_ratio: float = Field(0.6, gt=0, le=1)
    speech_s: float = Field(0.0, ge=0)
