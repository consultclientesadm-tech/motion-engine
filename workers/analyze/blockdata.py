"""Formato da análise por bloco (npz compacto) e resumo do bloco.

Uma linha por detecção: frame, track_id, caixa xyxy, 133 keypoints (x, y, score).
40 min x 30 fps x 3 pessoas ~ 216 mil linhas ~ 170 MB em float16, por isso um
arquivo por bloco: o render de um bloco só baixa o pedaço dele.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from workers.analyze.pose import N_KPTS


@dataclass
class BlockAnalysis:
    frames: np.ndarray     # (M,) int32  frame absoluto
    track_ids: np.ndarray  # (M,) int32
    boxes: np.ndarray      # (M, 4) float32
    kpts: np.ndarray       # (M, 133, 3) float16
    start_frame: int
    end_frame: int

    @classmethod
    def empty(cls, start_frame: int, end_frame: int) -> BlockAnalysis:
        return cls(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 4), np.float32),
                   np.zeros((0, N_KPTS, 3), np.float16), start_frame, end_frame)

    @classmethod
    def from_rows(cls, rows: list[tuple[int, int, np.ndarray, np.ndarray]],
                  start_frame: int, end_frame: int) -> BlockAnalysis:
        if not rows:
            return cls.empty(start_frame, end_frame)
        return cls(
            np.array([r[0] for r in rows], np.int32),
            np.array([r[1] for r in rows], np.int32),
            np.stack([r[2] for r in rows]).astype(np.float32),
            np.stack([r[3] for r in rows]).astype(np.float16),
            start_frame, end_frame,
        )

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, frames=self.frames, track_ids=self.track_ids, boxes=self.boxes,
                            kpts=self.kpts, range=np.array([self.start_frame, self.end_frame]))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> BlockAnalysis:
        z = np.load(io.BytesIO(data))
        s, e = (int(x) for x in z["range"])
        return cls(z["frames"], z["track_ids"], z["boxes"], z["kpts"], s, e)

    # ---- consultas -------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def people_per_frame(self) -> np.ndarray:
        counts = np.zeros(self.n_frames, np.int32)
        if len(self.frames):
            np.add.at(counts, self.frames - self.start_frame, 1)
        return counts

    def max_people(self) -> int:
        c = self.people_per_frame()
        return int(c.max()) if len(c) else 0

    def presence_ratio(self) -> float:
        c = self.people_per_frame()
        return float((c > 0).mean()) if len(c) else 0.0

    def tracks(self) -> list[int]:
        return sorted(set(int(t) for t in self.track_ids))

    def last_boxes(self) -> dict[int, np.ndarray]:
        """Última caixa de cada track (para costurar IDs numa retomada)."""
        out: dict[int, np.ndarray] = {}
        for f, t, b in zip(self.frames, self.track_ids, self.boxes):
            out[int(t)] = b
        return out

    def track_frames(self, track_id: int) -> np.ndarray:
        return self.frames[self.track_ids == track_id]

    def for_frame(self, frame: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
        idx = np.nonzero(self.frames == frame)[0]
        return [(int(self.track_ids[i]), self.boxes[i], self.kpts[i].astype(np.float32)) for i in idx]
