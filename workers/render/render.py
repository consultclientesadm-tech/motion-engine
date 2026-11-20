"""Render de um bloco de 30 s em 720p: Blender headless + composição FFmpeg."""
from __future__ import annotations

import json
from pathlib import Path

from settings import get_settings
from workers.media import ffmpeg, run
from workers.retarget.retarget import AVATAR_HEIGHT, retarget_block

BLENDER_SCRIPT = Path(__file__).with_name("blender_render.py")


def render_block(*, analysis, track_to_char: dict[int, str], avatars: dict[str, Path],
                 source_block: Path, src_w: int, src_h: int, fps: float,
                 workdir: Path, out: Path, engine: str | None = None) -> tuple[Path, dict]:
    """`source_block`: trecho original já em 720p. `analysis`: BlockAnalysis do bloco.
    Retorna (`out` mp4 720p sem áudio, pose retargetada usada pelo lipsync)."""
    s = get_settings()
    pose = retarget_block(analysis, track_to_char, src_w, src_h)
    used = {a["character"] for fr in pose["frames"] for a in fr["actors"]}

    if not used:  # ninguém mapeado neste bloco: devolve o original
        ffmpeg("-i", str(source_block), "-c:v", "libx264", "-preset", "medium", "-crf", "16",
               "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", "-an", str(out))
        return out, pose

    base = source_block
    if s.inpaint:  # apaga as pessoas que viram avatar (as outras ficam no quadro)
        from workers.inpaint.clean import clean_block
        base = workdir / "clean.mp4"
        clean_block(source_block=source_block, analysis=analysis, keep_tracks=set(track_to_char),
                    src_w=src_w, src_h=src_h, out=base)

    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "width": s.render_width, "height": s.render_height, "fps": fps,
        "avatar_height": AVATAR_HEIGHT, "engine": engine,
        "avatars": {c: str(p) for c, p in avatars.items() if c in used},
        "bones": pose["bones"], "rest_angles": pose["rest_angles"],
        "frames": pose["frames"], "out_dir": str(frames_dir),
    }
    cfg_path = workdir / "render.json"
    cfg_path.write_text(json.dumps(cfg))
    run([s.blender_bin, "-b", "--factory-startup", "--python", str(BLENDER_SCRIPT), "--", str(cfg_path)],
        timeout=6 * 3600)

    ffmpeg("-i", str(base),
           "-framerate", f"{fps}", "-start_number", "1", "-i", str(frames_dir / "frame_%04d.png"),
           "-filter_complex", f"[1:v]scale={s.render_width}:{s.render_height}[fg];"
                              f"[0:v][fg]overlay=0:0:format=auto:shortest=1,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-video_track_timescale", "90000", "-an", str(out))
    return out, pose
