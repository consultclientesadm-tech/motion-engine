"""Utilitários de mídia (ffprobe/ffmpeg) e medição de GPU-segundos."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from settings import get_settings


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], timeout: float | None = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise MediaError(f"{cmd[0]} falhou ({p.returncode}): {p.stderr[-2000:]}")
    return p.stdout


@dataclass
class Probe:
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool


def probe(path: str | Path) -> Probe:
    s = get_settings()
    out = json.loads(run([s.ffprobe_bin, "-v", "error", "-print_format", "json",
                          "-show_streams", "-show_format", str(path)]))
    v = next((st for st in out["streams"] if st["codec_type"] == "video"), None)
    if v is None:
        raise MediaError("arquivo sem trilha de vídeo")
    num, _, den = v.get("avg_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return Probe(
        duration_s=float(out["format"].get("duration") or v.get("duration") or 0),
        fps=round(fps, 3) or 30.0,
        width=int(v["width"]), height=int(v["height"]),
        has_audio=any(st["codec_type"] == "audio" for st in out["streams"]),
    )


def ffmpeg(*args: str, timeout: float | None = None) -> None:
    run([get_settings().ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", *args], timeout=timeout)


def normalize_source(src: Path, dst: Path, fps: float) -> None:
    """CFR + H.264 + AAC: frames e tempos batem entre análise, render e mux."""
    ffmpeg("-i", str(src), "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(dst))


def cut(src: Path, dst: Path, start_s: float, end_s: float, width: int | None = None,
        height: int | None = None, audio: bool = False) -> None:
    """Recorta [start, end) com precisão de frame (re-encode)."""
    vf = [f"scale={width}:{height}:flags=lanczos"] if width and height else []
    args = ["-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{end_s - start_s:.3f}"]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac"] if audio else ["-an"]
    ffmpeg(*args, str(dst))


def extract_audio(src: Path, dst: Path, sr: int = 48000) -> None:
    ffmpeg("-i", str(src), "-vn", "-ac", "2", "-ar", str(sr), str(dst))


@contextmanager
def workdir(job_id: str, name: str):
    base = Path(get_settings().work_dir) / job_id
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=base))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class GpuTimer:
    """Mede GPU-segundos de parede (a GPU alugada é cobrada por tempo, não por uso)."""

    def __enter__(self):
        self.t0 = time.monotonic()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.monotonic() - self.t0
        return False
