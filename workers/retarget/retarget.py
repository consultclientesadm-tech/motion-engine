"""Pose 2D (DWPose) -> rig do avatar, sem expressão facial.

Aproximação "boneco de papel": a câmera do Blender é ortográfica e olha
para +Y, então cada osso só gira no plano XZ (plano da imagem). O ângulo de
cada osso vem direto da direção 2D entre dois keypoints. É barato, estável e
bom o bastante para avatar aproximado; não reconstrói profundidade (braço
apontando para a câmera vira braço curto).

Saída por frame e por ator (coordenadas normalizadas 0..1 do quadro):
  root  [u, v]   ponto médio do quadril
  scale          altura do avatar relativa à altura do quadro
  angles {osso: graus no mundo}   (o script do Blender converte para local)
  face  [u1, v1, u2, v2] caixa do rosto, usada pelo lipsync
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# COCO: 0 nariz, 3/4 orelhas, 5/6 ombros, 7/8 cotovelos, 9/10 punhos,
# 11/12 quadris, 13/14 joelhos, 15/16 tornozelos. "L" = esquerda da pessoa.
SH_MID, HIP_MID = "sh_mid", "hip_mid"

# osso: (keypoint da cabeça do osso, keypoint da ponta, osso pai)
BONES: dict[str, tuple[int | str, int | str, str | None]] = {
    "spine": (HIP_MID, SH_MID, None),
    "head": (SH_MID, 0, "spine"),
    "upper_arm.L": (5, 7, "spine"),
    "forearm.L": (7, 9, "upper_arm.L"),
    "upper_arm.R": (6, 8, "spine"),
    "forearm.R": (8, 10, "upper_arm.R"),
    "thigh.L": (11, 13, None),
    "shin.L": (13, 15, "thigh.L"),
    "thigh.R": (12, 14, None),
    "shin.R": (14, 16, "thigh.R"),
}

# ângulos do rig em repouso (graus, a = atan2(x, z): 0 = para cima, 90 = para +X).
# "L" fica em +X (a pessoa encara a câmera). A-pose com braços a 45 graus.
REST_ANGLES = {
    "spine": 0.0, "head": 0.0,
    "upper_arm.L": 135.0, "forearm.L": 135.0,
    "upper_arm.R": -135.0, "forearm.R": -135.0,
    "thigh.L": 180.0, "shin.L": 180.0, "thigh.R": 180.0, "shin.R": 180.0,
}

AVATAR_HEIGHT = 1.75      # metros (o rig é normalizado para isso)
TORSO_FRAC = 0.30         # quadril->ombro ~ 30% da altura


def _kp(k: np.ndarray, ref: int | str, min_score: float) -> np.ndarray | None:
    if ref == SH_MID:
        a, b = _kp(k, 5, min_score), _kp(k, 6, min_score)
    elif ref == HIP_MID:
        a, b = _kp(k, 11, min_score), _kp(k, 12, min_score)
    else:
        return k[ref, :2] if k[ref, 2] >= min_score else None
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def image_angle(p0: np.ndarray, p1: np.ndarray) -> float:
    """Ângulo no mundo de um vetor de imagem (v cresce para baixo; z = -v)."""
    du, dv = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    return math.degrees(math.atan2(du, -dv))


def unwrap(prev: float | None, a: float) -> float:
    if prev is None:
        return a
    while a - prev > 180:
        a -= 360
    while a - prev < -180:
        a += 360
    return a


@dataclass
class ActorSmoother:
    """EMA por ator; segura o último valor quando o keypoint some."""
    alpha: float = 0.5
    angles: dict[str, float] = field(default_factory=dict)
    root: np.ndarray | None = None
    scale: float | None = None

    def angle(self, bone: str, a: float | None) -> float:
        prev = self.angles.get(bone)
        if a is None:
            return prev if prev is not None else REST_ANGLES[bone]
        a = unwrap(prev, a)
        v = a if prev is None else prev + self.alpha * (a - prev)
        self.angles[bone] = v
        return v

    def smooth(self, attr: str, value):
        prev = getattr(self, attr)
        v = value if prev is None else prev + self.alpha * (value - prev)
        setattr(self, attr, v)
        return v


def retarget_actor(kpts: np.ndarray, box: np.ndarray, frame_w: int, frame_h: int,
                   sm: ActorSmoother, min_score: float = 0.3) -> dict:
    angles = {}
    for bone, (a, b, _parent) in BONES.items():
        p0, p1 = _kp(kpts, a, min_score), _kp(kpts, b, min_score)
        # float(): os pontos vêm em np.float32 e o json.dumps do render não aceita numpy
        angles[bone] = round(float(sm.angle(bone, image_angle(p0, p1) if p0 is not None and p1 is not None else None)), 2)

    hip = _kp(kpts, HIP_MID, min_score)
    if hip is None:  # sem quadril visível: estima pelo meio-baixo da caixa
        hip = np.array([(box[0] + box[2]) / 2, box[1] + 0.55 * (box[3] - box[1])])
    sh = _kp(kpts, SH_MID, min_score)
    if sh is not None:
        height_px = float(np.linalg.norm(sh - hip)) / TORSO_FRAC
    else:
        height_px = float(box[3] - box[1])
    root = sm.smooth("root", np.array([hip[0] / frame_w, hip[1] / frame_h]))
    scale = sm.smooth("scale", height_px / frame_h)

    face = None
    nose = _kp(kpts, 0, min_score)
    if nose is not None:
        half = 0.07 * scale  # meio-lado da caixa do rosto, relativo à altura
        cu, cv = nose[0] / frame_w, nose[1] / frame_h
        aspect = frame_h / frame_w
        face = [round(float(cu - half * aspect), 4), round(float(cv - half * 1.2), 4),
                round(float(cu + half * aspect), 4), round(float(cv + half * 0.9), 4)]
    return {"root": [round(float(root[0]), 4), round(float(root[1]), 4)],
            "scale": round(float(scale), 4), "angles": angles, "face": face}


def retarget_block(analysis, track_to_char: dict[int, str], frame_w: int, frame_h: int) -> dict:
    """BlockAnalysis -> dicionário consumido pelo script do Blender."""
    smoothers: dict[int, ActorSmoother] = {}
    frames = []
    for f in range(analysis.start_frame, analysis.end_frame):
        actors, seen_chars = [], set()
        for track_id, box, kpts in analysis.for_frame(f):
            char = track_to_char.get(track_id)
            if char is None or char in seen_chars:
                continue  # track sem personagem (figurante) ou duplicado
            seen_chars.add(char)
            sm = smoothers.setdefault(track_id, ActorSmoother())
            actors.append({"track": track_id, "character": char,
                           **retarget_actor(kpts, box, frame_w, frame_h, sm)})
        frames.append({"f": f - analysis.start_frame, "actors": actors})
    return {"bones": {b: {"parent": p} for b, (_, _, p) in BONES.items()},
            "rest_angles": REST_ANGLES, "frames": frames}


def local_angles(world: dict[str, float]) -> dict[str, float]:
    """Mundo -> rotação local de cada osso (em torno do Y do mundo).
    local = (mundo_osso - mundo_pai) - (repouso_osso - repouso_pai)."""
    out = {}
    for bone, (_, _, parent) in BONES.items():
        wp = world[parent] if parent else 0.0
        rp = REST_ANGLES[parent] if parent else 0.0
        out[bone] = (world[bone] - wp) - (REST_ANGLES[bone] - rp)
    return out
