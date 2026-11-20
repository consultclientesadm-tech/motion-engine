"""Modelos SQLAlchemy. Estado do job, personagens, blocos (checkpoint) e gasto."""
from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utctoday() -> date:
    return utcnow().date()


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class JobState(str, enum.Enum):
    queued = "queued"
    analyzing = "analyzing"
    building = "building"
    rendering = "rendering"
    audio = "audio"
    post = "post"
    done = "done"
    paused_budget = "paused_budget"
    failed = "failed"
    cancelled = "cancelled"


# estados em que um worker está (ou pode estar) mexendo no job
ACTIVE_STATES = {JobState.queued, JobState.analyzing, JobState.building,
                 JobState.rendering, JobState.audio, JobState.post}


class BlockStage(str, enum.Enum):
    """Checkpoint por bloco. Cada valor significa 'esta etapa já terminou'."""
    pending = "pending"
    rendered = "rendered"      # 720p com avatares
    lipsynced = "lipsynced"    # boca sincronizada (ou pulado se não há voz)
    upscaled = "upscaled"      # 1080p pronto para o mux
    skipped = "skipped"        # sem pessoa: usa o trecho original


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    state: Mapped[JobState] = mapped_column(Enum(JobState), default=JobState.queued, index=True)
    video_url: Mapped[str] = mapped_column(String(1024))
    budget_usd: Mapped[float] = mapped_column(Float)
    daily_budget_usd: Mapped[float] = mapped_column(Float)
    target_resolution: Mapped[str] = mapped_column(String(8), default="1080p")
    process_only_with_people: Mapped[bool] = mapped_column(Boolean, default=True)

    duration_s: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    has_audio: Mapped[bool] = mapped_column(Boolean, default=True)

    source_key: Mapped[str | None] = mapped_column(String(512))   # cópia normalizada no S3
    tracks_key: Mapped[str | None] = mapped_column(String(512))
    poses_key: Mapped[str | None] = mapped_column(String(512))
    audio_key: Mapped[str | None] = mapped_column(String(512))    # trilha final (original ou dublada)
    output_key: Mapped[str | None] = mapped_column(String(512))

    estimated_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    script: Mapped[list | None] = mapped_column(JSON)   # falas opcionais para voz gerada
    error: Mapped[str | None] = mapped_column(Text)
    celery_task_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    characters: Mapped[list[Character]] = relationship(back_populates="job", order_by="Character.position",
                                                       cascade="all, delete-orphan")
    blocks: Mapped[list[Block]] = relationship(back_populates="job", order_by="Block.index",
                                               cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(String(128))
    sheet_url: Mapped[str] = mapped_column(String(1024))
    voice_ref_url: Mapped[str | None] = mapped_column(String(1024))
    voice_mode: Mapped[str] = mapped_column(String(16), default="clone")
    track_ids: Mapped[list] = mapped_column(JSON, default=list)   # mapa track -> personagem
    track_ids_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    avatar_key: Mapped[str | None] = mapped_column(String(512))   # .glb
    voice_key: Mapped[str | None] = mapped_column(String(512))    # .wav com as falas

    job: Mapped[Job] = relationship(back_populates="characters")


class Block(Base):
    """Bloco de 30 s: unidade de render, cobrança e retomada."""
    __tablename__ = "blocks"
    __table_args__ = (UniqueConstraint("job_id", "index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    has_people: Mapped[bool] = mapped_column(Boolean, default=True)
    people: Mapped[int] = mapped_column(Integer, default=0)       # máx. simultâneo no bloco
    stage: Mapped[BlockStage] = mapped_column(Enum(BlockStage), default=BlockStage.pending)
    analysis_key: Mapped[str | None] = mapped_column(String(512))  # npz de tracks+poses do bloco
    render_key: Mapped[str | None] = mapped_column(String(512))
    lipsync_key: Mapped[str | None] = mapped_column(String(512))
    final_key: Mapped[str | None] = mapped_column(String(512))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    job: Mapped[Job] = relationship(back_populates="blocks")

    @property
    def seconds(self) -> float:
        return self.end_s - self.start_s


class DailySpend(Base):
    """Uma linha por dia UTC, travada com SELECT ... FOR UPDATE ao cobrar."""
    __tablename__ = "daily_spend"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    usd: Mapped[float] = mapped_column(Float, default=0.0)


class Spend(Base):
    """Livro-razão de gasto: reserva (+) antes de cada etapa, acerto (+/-) depois."""
    __tablename__ = "spend"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    stage: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))   # reserve | settle
    usd: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
