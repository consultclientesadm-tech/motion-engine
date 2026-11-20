"""Concatena os blocos finais e junta com a trilha de áudio (FFmpeg)."""
from __future__ import annotations

from pathlib import Path

from workers.media import ffmpeg


def concat_list(paths: list[Path], list_file: Path) -> Path:
    list_file.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths))
    return list_file


def mux(blocks: list[Path], audio: Path | None, out: Path, workdir: Path) -> Path:
    """Todos os blocos têm o mesmo codec/resolução/fps, então o concat é sem re-encode."""
    lst = concat_list(blocks, workdir / "blocks.txt")
    video = workdir / "video.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video))
    if audio is None:
        ffmpeg("-i", str(video), "-c", "copy", "-movflags", "+faststart", str(out))
        return out
    ffmpeg("-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", str(out))
    return out


def mix_dub(original_audio: Path | None, voices: list[Path], out: Path, bed_db: float = -14.0) -> Path:
    """Vozes geradas por cima do áudio original abaixado (`bed_db`).
    Não separa a fala original da música; para isso plugue um Demucs aqui."""
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    idx = 0
    if original_audio is not None:
        inputs += ["-i", str(original_audio)]
        filters.append(f"[0:a]volume={bed_db}dB[bed]")
        labels.append("[bed]")
        idx = 1
    for i, v in enumerate(voices):
        inputs += ["-i", str(v)]
        filters.append(f"[{idx + i}:a]aresample=48000,aformat=channel_layouts=stereo[v{i}]")
        labels.append(f"[v{i}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest[out]")
    ffmpeg(*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "48000", str(out))
    return out
