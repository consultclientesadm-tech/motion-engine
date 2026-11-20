"""Tracking multi-pessoa: YOLO (classe pessoa) + ByteTrack via Ultralytics.

A análise corre em sequência pelo vídeo (o ByteTrack precisa de continuidade),
mas grava um arquivo por bloco de 30 s. Se o worker cair, a retomada começa no
primeiro bloco sem arquivo e "costura" os IDs novos aos antigos por IoU.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from settings import get_settings


@dataclass
class FrameTracks:
    frame: int
    ids: np.ndarray        # (N,) int
    boxes: np.ndarray      # (N, 4) xyxy float
    conf: np.ndarray       # (N,)


@dataclass
class Tracker:
    """Embrulha o YOLO+ByteTrack e mantém o mapa de IDs após uma retomada."""
    weights: str = field(default_factory=lambda: get_settings().yolo_weights)
    device: str = field(default_factory=lambda: get_settings().device)
    conf: float = 0.35
    min_box_frac: float = 0.02     # ignora pessoas minúsculas (fundo/plateia)
    id_offset: int = 0
    _remap: dict[int, int] = field(default_factory=dict)
    _model: object = None

    def load(self):
        if self._model is None:
            from ultralytics import YOLO  # import pesado: só no worker de GPU
            self._model = YOLO(self.weights)
        return self._model

    def stitch(self, previous_last: dict[int, np.ndarray], first: FrameTracks,
               thr: float = 0.5) -> dict[int, int]:
        """Retomada: liga os IDs do tracker novo aos IDs antigos por IoU.
        `first.ids` já vem mapeado; devolve {id_mapeado: id_antigo} e atualiza
        o mapa interno para os próximos frames."""
        raw_of = {v: k for k, v in self._remap.items()}
        used: set[int] = set()
        out: dict[int, int] = {}
        for i, mapped in enumerate(first.ids):
            best, best_iou = None, thr
            for old_id, old_box in previous_last.items():
                if old_id in used:
                    continue
                iou = box_iou(first.boxes[i], old_box)
                if iou > best_iou:
                    best, best_iou = old_id, iou
            if best is not None:
                used.add(best)
                out[int(mapped)] = best
                if int(mapped) in raw_of:
                    self._remap[raw_of[int(mapped)]] = best
        return out

    def _map(self, raw_id: int) -> int:
        if raw_id not in self._remap:
            self._remap[raw_id] = raw_id + self.id_offset
        return self._remap[raw_id]

    def step(self, frame_idx: int, image: np.ndarray) -> FrameTracks:
        model = self.load()
        h, w = image.shape[:2]
        r = model.track(image, persist=True, tracker="bytetrack.yaml", classes=[0],
                        conf=self.conf, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            return FrameTracks(frame_idx, np.zeros(0, int), np.zeros((0, 4), np.float32), np.zeros(0))
        boxes = b.xyxy.cpu().numpy().astype(np.float32)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(w * h)
        keep = area >= self.min_box_frac
        ids = np.array([self._map(int(i)) for i in b.id.cpu().numpy()[keep]], dtype=int)
        return FrameTracks(frame_idx, ids, boxes[keep], b.conf.cpu().numpy()[keep])


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def read_frames(video: Path, start_frame: int, end_frame: int) -> Iterator[tuple[int, np.ndarray]]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for idx in range(start_frame, end_frame):
            ok, img = cap.read()
            if not ok:
                break
            yield idx, img
    finally:
        cap.release()


def run_tracking(video: Path, start_frame: int, end_frame: int,
                 tracker: Tracker) -> Iterator[tuple[np.ndarray, FrameTracks]]:
    """Itera (imagem, tracks) de [start_frame, end_frame)."""
    for idx, img in read_frames(video, start_frame, end_frame):
        yield img, tracker.step(idx, img)
