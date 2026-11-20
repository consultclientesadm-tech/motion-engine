"""Apaga a pessoa original do trecho antes de desenhar o avatar (inpainting).

Sem isso, onde o avatar não cobre a pessoa de verdade aparece atrás dele: era a
"borra" do primeiro teste. Por bloco de 30 s:

1. máscara das pessoas a trocar (YOLO-seg; só as que viram personagem, casadas
   com as caixas da análise por IoU), dilatada e com borda suave;
2. câmera parada: fundo = mediana temporal dos pixels sem pessoa (amostra de
   frames). O que nunca aparece sem pessoa no bloco é preenchido uma vez só
   com LaMa no próprio fundo. Barato e sem tremido;
3. câmera mexendo: LaMa quadro a quadro (pode tremer um pouco na borda;
   o avatar por cima esconde a maior parte).

Sem o peso do LaMa, cai no cv2.inpaint (pior, mas não quebra o job).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from settings import get_settings

log = logging.getLogger(__name__)


# ---- máscaras ---------------------------------------------------------------------

def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def box_mask(boxes: np.ndarray, h: int, w: int, pad: float = 0.04) -> np.ndarray:
    """Máscara retangular (reserva quando o seg não acha a pessoa)."""
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        m[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))] = 255
    return m


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2
    if px <= 0 or not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    """0..1 float com borda suave (para a costura não ficar marcada)."""
    import cv2
    m = mask.astype(np.float32) / 255.0
    if px > 0:
        m = cv2.GaussianBlur(m, (0, 0), px / 2.0)
        m = np.maximum(m, mask.astype(np.float32) / 255.0)
    return m[..., None]


@dataclass
class Segmenter:
    weights: str = field(default_factory=lambda: get_settings().yolo_seg_weights)
    device: str = field(default_factory=lambda: get_settings().device)
    _model: object = None

    def load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def masks(self, image: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Une as máscaras das pessoas que casam com `targets` (caixas xyxy na
        escala de `image`). Alvo sem máscara correspondente vira retângulo."""
        import cv2
        h, w = image.shape[:2]
        out = np.zeros((h, w), np.uint8)
        if len(targets) == 0:
            return out
        r = self.load()(image, classes=[0], conf=0.25, device=self.device, verbose=False, retina_masks=True)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        polys = r.masks.data.cpu().numpy() if r.masks is not None else np.zeros((0, h, w))
        for t in targets:
            best, best_iou = -1, 0.3
            for i, b in enumerate(boxes):
                iou = box_iou(t, b)
                if iou > best_iou:
                    best, best_iou = i, iou
            if best >= 0 and best < len(polys):
                m = polys[best]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                out[m > 0.5] = 255
            else:
                out |= box_mask(np.array([t]), h, w)
        return out


# ---- preenchimento -------------------------------------------------------------------

@dataclass
class Lama:
    """big-lama em TorchScript (o mesmo arquivo do simple-lama-inpainting)."""
    weights: str = field(default_factory=lambda: get_settings().lama_weights)
    device: str = field(default_factory=lambda: get_settings().device)
    _model: object = None

    def available(self) -> bool:
        return Path(self.weights).exists()

    def load(self):
        if self._model is None:
            import torch
            dev = self.device if torch.cuda.is_available() else "cpu"
            self._model = torch.jit.load(self.weights, map_location=dev).eval()
            self._dev = dev
        return self._model

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image BGR uint8, mask uint8 (255 = preencher). Devolve BGR uint8."""
        if not mask.any():
            return image
        if not self.available():
            import cv2
            return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        import torch
        model = self.load()
        h, w = image.shape[:2]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img = np.pad(image[..., ::-1], ((0, ph), (0, pw), (0, 0)), mode="reflect")
        msk = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        with torch.inference_mode():
            t_img = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float().div(255).to(self._dev)
            t_msk = (torch.from_numpy(msk)[None, None].float() > 127).float().to(self._dev)
            out = model(t_img, t_msk)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        return np.ascontiguousarray(out[:h, :w, ::-1])


def camera_motion(samples: list[np.ndarray], masks: list[np.ndarray]) -> float:
    """Deslocamento médio (px) entre amostras vizinhas, fora das pessoas."""
    import cv2
    if len(samples) < 2:
        return 0.0
    shifts = []
    win = None
    for a, b, ma, mb in zip(samples, samples[1:], masks, masks[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        keep = ((ma | mb) == 0).astype(np.float32)
        if keep.mean() < 0.2:
            continue
        mean = ga[keep > 0].mean()
        ga, gb = ga * keep + mean * (1 - keep), gb * keep + mean * (1 - keep)
        if win is None:
            win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win)
        shifts.append(float(np.hypot(dx, dy)))
    return float(np.median(shifts)) if shifts else 0.0


def temporal_background(samples: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mediana por pixel só dos frames em que o pixel está livre.
    Devolve (fundo BGR uint8, buraco uint8 = pixels nunca livres)."""
    stack = np.stack(samples)                         # (N, H, W, 3)
    free = np.stack(masks) == 0                       # (N, H, W)
    h, w = free.shape[1:]
    bg = np.zeros((h, w, 3), np.uint8)
    for y0 in range(0, h, 90):                        # em faixas, para não estourar memória
        sl = slice(y0, y0 + 90)
        s = stack[:, sl].astype(np.float32)
        s[~free[:, sl]] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(s, axis=0)
        bg[sl] = np.nan_to_num(med, nan=0).clip(0, 255).astype(np.uint8)
    hole = (~free.any(axis=0)).astype(np.uint8) * 255
    return bg, hole


# ---- bloco -----------------------------------------------------------------------------

@dataclass
class CleanStats:
    method: str
    motion_px: float
    masked_ratio: float


def clean_block(*, source_block: Path, analysis, keep_tracks: set[int], src_w: int, src_h: int,
                out: Path, segmenter: Segmenter | None = None, lama: Lama | None = None,
                sample_every: int | None = None) -> CleanStats:
    """Escreve `out` (mp4 sem áudio, mesma resolução de `source_block`) sem as
    pessoas de `keep_tracks`. As outras pessoas (fundo, plateia) ficam."""
    import cv2
    s = get_settings()
    seg = segmenter or Segmenter()
    lama = lama or Lama()
    sample_every = sample_every or s.inpaint_sample_every

    cap = cv2.VideoCapture(str(source_block))
    fps = cap.get(cv2.CAP_PROP_FPS) or s.fps
    frames: list[np.ndarray] = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    if not frames:
        raise RuntimeError(f"bloco vazio: {source_block}")
    h, w = frames[0].shape[:2]
    sx, sy = w / float(src_w), h / float(src_h)

    # caixas das pessoas a trocar, por frame do bloco, na escala do bloco
    per_frame: list[np.ndarray] = [np.zeros((0, 4), np.float32) for _ in frames]
    for f, t, b in zip(analysis.frames, analysis.track_ids, analysis.boxes):
        i = int(f) - analysis.start_frame
        if 0 <= i < len(frames) and int(t) in keep_tracks:
            per_frame[i] = np.vstack([per_frame[i], (b * [sx, sy, sx, sy])[None].astype(np.float32)])

    grow_px = max(4, int(round(h * s.inpaint_grow)))
    masks = [grow(seg.masks(img, per_frame[i]), grow_px) if len(per_frame[i]) else np.zeros((h, w), np.uint8)
             for i, img in enumerate(frames)]
    masked_ratio = float(np.mean([m.mean() / 255 for m in masks]))
    if masked_ratio == 0:
        _write(frames, fps, out)
        return CleanStats("none", 0.0, 0.0)

    idx = list(range(0, len(frames), sample_every))
    motion = camera_motion([frames[i] for i in idx], [masks[i] for i in idx])
    fb = max(2, grow_px // 2)

    if motion <= s.inpaint_static_px:
        bg, hole = temporal_background([frames[i] for i in idx], [masks[i] for i in idx])
        if hole.any():
            bg = lama(bg, grow(hole, 2))
        result = []
        for img, m in zip(frames, masks):
            if not m.any():
                result.append(img)
                continue
            a = feather(m, fb)
            result.append((img * (1 - a) + bg * a).astype(np.uint8))
        method = "temporal"
    else:
        result = []
        for img, m in zip(frames, masks):
            if not m.any():
                result.append(img)
                continue
            filled = lama(img, m)
            a = feather(m, fb)
            result.append((img * (1 - a) + filled * a).astype(np.uint8))
        method = "lama" if lama.available() else "opencv"
    _write(result, fps, out)
    log.info("inpaint %s: método=%s movimento=%.1fpx área=%.1f%%", source_block.name, method, motion,
             masked_ratio * 100)
    return CleanStats(method, motion, masked_ratio)


def _write(frames: list[np.ndarray], fps: float, out: Path) -> None:
    """Grava via ffmpeg (h264 de verdade, timescale igual aos outros blocos)."""
    import subprocess
    s = get_settings()
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([s.ffmpeg_bin, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "14", "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", "-an",
                          str(out)], stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
    finally:
        p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg falhou ao gravar o bloco limpo")
