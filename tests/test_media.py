"""FFmpeg de ponta a ponta (sem GPU): normaliza, corta, passthrough, concat + mux."""
import shutil

import pytest

from workers.media import cut, extract_audio, ffmpeg, normalize_source, probe
from workers.post.mux import mix_dub, mux
from workers.post.upscale import passthrough

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg ausente")


def test_blocks_concat_and_mux(tmp_path):
    raw = tmp_path / "raw.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=5",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-shortest",
           "-c:v", "libx264", "-c:a", "aac", str(raw))
    src = tmp_path / "src.mp4"
    normalize_source(raw, src, 30)
    info = probe(src)
    assert info.fps == 30 and info.has_audio and 4.9 < info.duration_s < 5.2

    # bloco "renderizado" (corte 720p) + bloco sem pessoa (passthrough 720p)
    b0 = tmp_path / "b0.mp4"
    cut(src, tmp_path / "c0.mp4", 0, 2.5, 1280, 720)
    passthrough(tmp_path / "c0.mp4", b0, 0, 2.5, 1280, 720, 30)
    b1 = passthrough(src, tmp_path / "b1.mp4", 2.5, 5.0, 1280, 720, 30)
    wav = tmp_path / "a.wav"
    extract_audio(src, wav)
    dub = mix_dub(wav, [wav], tmp_path / "dub.wav")
    out = mux([b0, b1], dub, tmp_path / "out.mp4", tmp_path)
    o = probe(out)
    assert (o.width, o.height) == (1280, 720) and o.has_audio
    assert abs(o.duration_s - 5.0) < 0.2
