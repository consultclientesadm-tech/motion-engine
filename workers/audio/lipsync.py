"""Lipsync com Wav2Lip no rosto renderizado.

As caixas do rosto vêm do retarget (nariz -> caixa), então não rodamos o
detector de rosto do Wav2Lip (que falha em rosto de avatar). Só os frames em
que o personagem está falando (energia > limiar na trilha dele) são alterados.
SadTalker pode entrar no lugar implementando `LipSyncer.infer_batch`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from settings import get_settings

IMG = 96
MEL_STEP = 16
SR = 16000


class LipSyncer:
    def __init__(self):
        self._model = None
        self._audio = None
        self.device = None

    def load(self):
        if self._model is None:
            import torch
            s = get_settings()
            sys.path.insert(0, s.wav2lip_repo)
            import audio as w2l_audio  # do repositório Wav2Lip
            from models import Wav2Lip
            self.device = s.device if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(s.wav2lip_checkpoint, map_location=self.device)
            state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
            m = Wav2Lip()
            m.load_state_dict(state)
            self._model = m.to(self.device).eval()
            self._audio = w2l_audio
        return self._model

    def mel_chunks(self, wav16k: np.ndarray, fps: float, n_frames: int) -> list[np.ndarray]:
        self.load()
        mel = self._audio.melspectrogram(wav16k)
        mult = 80.0 / fps
        chunks = []
        for i in range(n_frames):
            start = int(i * mult)
            if start + MEL_STEP > mel.shape[1]:
                chunks.append(mel[:, -MEL_STEP:] if mel.shape[1] >= MEL_STEP
                              else np.pad(mel, ((0, 0), (0, MEL_STEP - mel.shape[1]))))
            else:
                chunks.append(mel[:, start:start + MEL_STEP])
        return chunks

    def infer_batch(self, faces: list[np.ndarray], mels: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        model = self.load()
        img = np.asarray(faces, np.float32)
        masked = img.copy()
        masked[:, IMG // 2:] = 0
        x = np.concatenate([masked, img], axis=3) / 255.0
        m = np.asarray(mels, np.float32)[:, None]
        with torch.no_grad():
            pred = model(torch.from_numpy(m).to(self.device),
                         torch.from_numpy(x.transpose(0, 3, 1, 2)).to(self.device))
        return list((pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8))


def speaking(wav16k: np.ndarray, fps: float, n_frames: int, thr: float = 0.01) -> np.ndarray:
    hop = SR / fps
    out = np.zeros(n_frames, bool)
    for i in range(n_frames):
        seg = wav16k[int(i * hop):int((i + 1) * hop)]
        out[i] = len(seg) > 0 and float(np.sqrt(np.mean(seg ** 2))) > thr
    return out


def _paste(frame: np.ndarray, patch: np.ndarray, box: tuple[int, int, int, int]) -> None:
    import cv2
    x1, y1, x2, y2 = box
    patch = cv2.resize(patch, (x2 - x1, y2 - y1))
    h, w = patch.shape[:2]
    mask = np.zeros((h, w), np.float32)
    mask[h // 2:, :] = 1.0  # só a metade de baixo (boca/queixo)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1, w // 12))[..., None]
    region = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (patch * mask + region * (1 - mask)).astype(np.uint8)


def lipsync_block(*, video: Path, pose: dict, voices: dict[str, np.ndarray], fps: float,
                  out: Path, syncer: LipSyncer, batch: int = 64) -> Path:
    """`voices`: personagem -> trilha 16 kHz recortada para o bloco."""
    import cv2
    s = get_settings()
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    n = len(frames)

    for char, wav in voices.items():
        talk = speaking(wav, fps, n)
        mels = syncer.mel_chunks(wav, fps, n)
        jobs: list[tuple[int, tuple[int, int, int, int]]] = []
        for fr in pose["frames"][:n]:
            a = next((a for a in fr["actors"] if a["character"] == char and a.get("face")), None)
            if a is None or not talk[fr["f"]]:
                continue
            u1, v1, u2, v2 = a["face"]
            box = (max(0, int(u1 * W)), max(0, int(v1 * H)), min(W, int(u2 * W)), min(H, int(v2 * H)))
            if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
                jobs.append((fr["f"], box))
        for i in range(0, len(jobs), batch):
            part = jobs[i:i + batch]
            faces = [cv2.resize(frames[f][b[1]:b[3], b[0]:b[2]], (IMG, IMG)) for f, b in part]
            preds = syncer.infer_batch(faces, [mels[f] for f, _ in part])
            for (f, b), p in zip(part, preds):
                _paste(frames[f], p, b)

    enc = subprocess.Popen([s.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-video_track_timescale", "90000", str(out)], stdin=subprocess.PIPE)
    for img in frames:
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg falhou ao gravar o bloco com lipsync")
    return out
