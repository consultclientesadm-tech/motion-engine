"""Configuração central, lida do ambiente / .env (mesmos nomes do .env.example)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Infra
    database_url: str = "postgresql+psycopg://motion:motion@localhost:5432/motion"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint: str | None = None          # R2/MinIO; vazio = AWS S3
    s3_bucket: str = "motion-engine"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "auto"
    s3_tmp_retention_days: int = 7          # lifecycle de tmp/ (blocos, faces, tracks)
    work_dir: str = "/tmp/motion-engine"

    # Dinheiro
    gpu_usd_per_hour: float = 1.2           # Vast.ai A100
    cpu_usd_per_hour: float = 0.05
    daily_budget_usd: float = 90.0          # teto diário do servidor (manda sobre o do request)
    max_job_budget_usd: float = 200.0

    # Render
    block_seconds: float = 30.0
    render_width: int = 1280
    render_height: int = 720
    output_width: int = 1920
    output_height: int = 1080
    fps: float = 30.0
    min_person_ratio: float = 0.05          # bloco "tem pessoa" se >=5% dos frames tiverem alguém

    # Ferramentas / pesos
    blender_bin: str = "blender"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    device: str = "cuda"
    yolo_weights: str = "yolov8m.pt"
    dwpose_det_onnx: str = "models/yolox_l.onnx"
    dwpose_pose_onnx: str = "models/dw-ll_ucoco_384.onnx"
    triposr_model: str = "stabilityai/TripoSR"
    realesrgan_weights: str = "models/RealESRGAN_x2plus.pth"
    wav2lip_checkpoint: str = "models/wav2lip_gan.pth"
    wav2lip_repo: str = "/opt/Wav2Lip"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
