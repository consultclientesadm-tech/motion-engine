"""Mapa automático track -> personagem (o usuário pode sobrescrever pela API).

O ByteTrack quebra o ID de uma mesma pessoa depois de oclusão ou corte. Sem
re-identificação por aparência, usamos continuidade espaço-temporal:
  * tracks curtos (< min_frames) são ignorados (ruído/figurante de passagem);
  * os primeiros tracks simultâneos viram os personagens, da esquerda p/ direita;
  * cada track novo herda o personagem livre (sem track ativo naquele momento)
    cujo último track terminou mais perto, no tempo e no espaço.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackSummary:
    id: int
    first: int
    last: int
    count: int
    first_box: list[float]
    last_box: list[float]

    @property
    def cx_first(self) -> float:
        return (self.first_box[0] + self.first_box[2]) / 2

    @property
    def cx_last(self) -> float:
        return (self.last_box[0] + self.last_box[2]) / 2


class SummaryBuilder:
    def __init__(self):
        self.tracks: dict[int, TrackSummary] = {}

    def add(self, frame: int, track_id: int, box) -> None:
        b = [float(x) for x in box]
        t = self.tracks.get(track_id)
        if t is None:
            self.tracks[track_id] = TrackSummary(track_id, frame, frame, 1, b, b)
        else:
            t.last, t.count, t.last_box = frame, t.count + 1, b

    def to_json(self) -> list[dict]:
        return [t.__dict__ for t in sorted(self.tracks.values(), key=lambda t: t.first)]

    @classmethod
    def from_json(cls, rows: list[dict]) -> SummaryBuilder:
        sb = cls()
        for r in rows:
            sb.tracks[r["id"]] = TrackSummary(**r)
        return sb


def auto_assign(tracks: list[TrackSummary], characters: list[str], frame_w: float,
                min_frames: int = 30, max_gap_frames: int | None = None) -> dict[str, list[int]]:
    tracks = sorted((t for t in tracks if t.count >= min_frames), key=lambda t: t.first)
    out: dict[str, list[int]] = {c: [] for c in characters}
    last: dict[str, TrackSummary] = {}

    # semente: tracks que se sobrepõem ao primeiro, ordenados por x
    if tracks:
        t0 = tracks[0]
        seed = sorted([t for t in tracks if t.first <= t0.last][: len(characters)], key=lambda t: t.cx_first)
        for c, t in zip(characters, seed):
            out[c].append(t.id)
            last[c] = t
        tracks = [t for t in tracks if t not in seed]

    for t in tracks:
        free = [c for c in characters if c not in last or last[c].last < t.first]
        if not free:
            continue  # todos os personagens já estão em cena: é figurante
        def cost(c: str) -> float:
            if c not in last:
                return 0.5  # personagem ainda sem track: custo neutro
            gap = t.first - last[c].last
            if max_gap_frames is not None and gap > max_gap_frames:
                return 10.0
            return abs(t.cx_first - last[c].cx_last) / frame_w + gap / 3000.0
        best = min(free, key=cost)
        out[best].append(t.id)
        last[best] = t
    return out
