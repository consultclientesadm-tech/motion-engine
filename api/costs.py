"""Modelo de custo único, usado em três lugares:
  * na API, para recusar o job antes de gastar (estimativa > teto);
  * nos workers, para reservar o custo de cada etapa/bloco antes de rodar;
  * em scripts/cost_estimate.py.

Os segundos de GPU por frame foram medidos numa RTX A5000 (RunPod, 20/11/2026,
render 720p) e ajustados para o render padrão de 960x540. Recalibre no seu
hardware (veja README, seção Calibração).
Todos podem ser sobrescritos por variável de ambiente COST_<NOME>.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field

from settings import Settings, get_settings


def _f(name: str, default: float) -> float:
    return float(os.environ.get(f"COST_{name.upper()}", default))


# GPU-segundos por frame
DETECT_S = _f("detect_s", 0.012)            # YOLOv8m + ByteTrack, todo frame
POSE_S_PER_PERSON = _f("pose_s", 0.008)     # DWPose por pessoa, frames com gente
RENDER_S_BASE = _f("render_s_base", 0.15)   # Blender EEVEE 540p, cena (720p medido: ~0,21)
RENDER_S_PER_PERSON = _f("render_s_person", 0.10)
LIPSYNC_S_PER_FACE = _f("lipsync_s", 0.03)  # Wav2Lip por rosto que fala
UPSCALE_S = _f("upscale_s", 0.35)           # Real-ESRGAN x2 de 960x540 (de 1280x720 medido: 0,55)
INPAINT_S = _f("inpaint_s", 0.03)           # YOLO-seg + composição do fundo, frames com gente
# GPU-segundos fixos
AVATAR_S_PER_CHAR = _f("avatar_s", 50.0)    # TripoSR + textura + auto-rig no Blender
TTS_S_PER_AUDIO_S = _f("tts_s", 0.3)        # XTTS/F5 por segundo de fala
TASK_OVERHEAD_S = _f("overhead_s", 20.0)    # carregar pesos por tarefa
# CPU
MUX_CPU_S_PER_VIDEO_S = _f("mux_cpu_s", 0.6)


@dataclass
class Estimate:
    duration_s: float
    people: int
    people_ratio: float
    blocks: int
    blocks_with_people: int
    gpu_seconds: dict[str, float] = field(default_factory=dict)
    usd: dict[str, float] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return round(sum(self.usd.values()), 4)

    @property
    def gpu_hours(self) -> float:
        return round(sum(self.gpu_seconds.values()) / 3600.0, 3)

    def to_dict(self) -> dict:
        return {**asdict(self), "total_usd": self.total_usd, "gpu_hours": self.gpu_hours}


def gpu_usd(seconds: float, s: Settings | None = None) -> float:
    s = s or get_settings()
    return round(seconds * s.gpu_usd_per_hour / 3600.0, 6)


def cpu_usd(seconds: float, s: Settings | None = None) -> float:
    s = s or get_settings()
    return round(seconds * s.cpu_usd_per_hour / 3600.0, 6)


# ---- custo por etapa (GPU-segundos) -------------------------------------

def analyze_s(duration_s: float, fps: float, people: int, people_ratio: float) -> float:
    frames = duration_s * fps
    return frames * DETECT_S + frames * people_ratio * POSE_S_PER_PERSON * people + TASK_OVERHEAD_S


def build_s(characters: int, speech_s: float = 0.0) -> float:
    return characters * AVATAR_S_PER_CHAR + speech_s * TTS_S_PER_AUDIO_S + TASK_OVERHEAD_S


def render_block_s(seconds: float, fps: float, people: int) -> float:
    inpaint = INPAINT_S if get_settings().inpaint else 0.0
    return seconds * fps * (RENDER_S_BASE + RENDER_S_PER_PERSON * max(people, 1) + inpaint) + TASK_OVERHEAD_S


def lipsync_block_s(seconds: float, fps: float, faces: int) -> float:
    if faces <= 0:
        return 0.0
    return seconds * fps * LIPSYNC_S_PER_FACE * faces + TASK_OVERHEAD_S


def upscale_block_s(seconds: float, fps: float) -> float:
    return seconds * fps * UPSCALE_S + TASK_OVERHEAD_S


def mux_cpu_s(duration_s: float) -> float:
    return duration_s * MUX_CPU_S_PER_VIDEO_S


# ---- estimativa do job inteiro ------------------------------------------

def estimate(duration_s: float, people: int = 3, *, characters: int | None = None,
             fps: float = 30.0, target_resolution: str = "1080p",
             only_people: bool = True, people_ratio: float = 0.6,
             speech_s: float = 0.0, lipsync: bool = True,
             s: Settings | None = None) -> Estimate:
    """Estimativa antes de rodar. `people_ratio` = fração do vídeo com gente
    (antes da análise não sabemos; depois dela os workers usam o valor real)."""
    s = s or get_settings()
    characters = people if characters is None else characters
    ratio = people_ratio if only_people else 1.0
    n_blocks = max(1, math.ceil(duration_s / s.block_seconds))
    n_people_blocks = max(1, round(n_blocks * ratio))
    rendered_s = min(duration_s, n_people_blocks * s.block_seconds)
    blocks_rendered = n_people_blocks

    est = Estimate(duration_s=duration_s, people=people, people_ratio=ratio,
                   blocks=n_blocks, blocks_with_people=n_people_blocks)
    g = est.gpu_seconds
    g["analyze"] = analyze_s(duration_s, fps, people, ratio)
    g["build"] = build_s(characters, speech_s)
    # cada bloco paga o overhead de carga separadamente
    per_block = rendered_s / blocks_rendered
    g["render"] = blocks_rendered * render_block_s(per_block, fps, people)
    g["lipsync"] = blocks_rendered * lipsync_block_s(per_block, fps, people) if lipsync else 0.0
    g["upscale"] = (blocks_rendered * upscale_block_s(per_block, fps)
                    if target_resolution == "1080p" else 0.0)
    for k, v in g.items():
        g[k] = round(v, 1)
        est.usd[k] = gpu_usd(v, s)
    est.usd["mux"] = cpu_usd(mux_cpu_s(duration_s), s)
    return est
