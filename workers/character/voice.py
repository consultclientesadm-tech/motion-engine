"""Voz por personagem: XTTS v2 (clonagem) ou F5-TTS / XTTS (geração).

Só roda quando o job traz `script` (falas com tempo). Sem script, o job mantém
o áudio original e o lipsync é pulado.

  voice_mode="clone"    -> XTTS v2 com voice_ref_url como speaker_wav.
  voice_mode="generate" -> F5-TTS se houver voice_ref (estilo de referência);
                           senão, uma voz de estúdio do XTTS escolhida pelo nome.

Saída: uma trilha mono 24 kHz do tamanho do vídeo com as falas posicionadas.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from settings import get_settings

SR = 24000
XTTS_STUDIO_VOICES = ["Ana Florence", "Claribel Dervla", "Daisy Studious", "Gracie Wise",
                      "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min"]


class VoiceEngine:
    def __init__(self, language: str = "pt"):
        self.language = language
        self._xtts = None
        self._f5 = None

    def xtts(self):
        if self._xtts is None:
            from TTS.api import TTS
            s = get_settings()
            self._xtts = TTS(s.xtts_model).to(s.device)
        return self._xtts

    def f5(self):
        if self._f5 is None:
            from f5_tts.api import F5TTS
            self._f5 = F5TTS(device=get_settings().device)
        return self._f5

    def say(self, text: str, mode: str, name: str, ref_wav: Path | None) -> np.ndarray:
        if mode == "clone":
            if ref_wav is None:
                raise ValueError(f"personagem {name}: voice_mode=clone precisa de voice_ref_url")
            wav = self.xtts().tts(text=text, speaker_wav=str(ref_wav), language=self.language)
            return _resample(np.asarray(wav, np.float32), self.xtts().synthesizer.output_sample_rate)
        if ref_wav is not None:
            wav, sr, _ = self.f5().infer(ref_file=str(ref_wav), ref_text="", gen_text=text)
            return _resample(np.asarray(wav, np.float32), sr)
        speaker = XTTS_STUDIO_VOICES[int(hashlib.md5(name.encode()).hexdigest(), 16) % len(XTTS_STUDIO_VOICES)]
        wav = self.xtts().tts(text=text, speaker=speaker, language=self.language)
        return _resample(np.asarray(wav, np.float32), self.xtts().synthesizer.output_sample_rate)


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return wav
    n = int(round(len(wav) * SR / sr))
    return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


def place_lines(clips: list[tuple[float, np.ndarray]], duration_s: float) -> np.ndarray:
    """Posiciona cada fala em start_s; se encostar na anterior, empurra para depois dela."""
    track = np.zeros(int(duration_s * SR) + SR, np.float32)
    cursor = 0
    for start_s, wav in sorted(clips, key=lambda c: c[0]):
        i = max(int(start_s * SR), cursor)
        j = min(i + len(wav), len(track))
        track[i:j] += wav[: j - i]
        cursor = j + int(0.05 * SR)
    return track[: int(duration_s * SR)]


def build_voice_track(engine: VoiceEngine, character: dict, lines: list[dict],
                      duration_s: float, ref_wav: Path | None) -> tuple[np.ndarray, float]:
    """-> (trilha, segundos de fala). `lines` já filtradas para o personagem."""
    clips = [(ln["start_s"], engine.say(ln["text"], character["voice_mode"], character["name"], ref_wav))
             for ln in lines]
    speech_s = sum(len(w) for _, w in clips) / SR
    return place_lines(clips, duration_s), speech_s


def write_wav(path: Path, wav: np.ndarray, sr: int = SR) -> Path:
    import soundfile as sf
    sf.write(str(path), np.clip(wav, -1, 1), sr)
    return path
