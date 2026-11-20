"""DWPose (corpo inteiro, 133 pontos COCO-WholeBody) sobre as caixas do tracker.

Usa o rtmlib (ONNX Runtime), que roda o DWPose sem mmcv/mmpose. Recebe as
caixas do ByteTrack, então não roda um segundo detector.

Índices úteis (COCO-WholeBody): 0-16 corpo, 17-22 pés, 23-90 rosto (68),
91-111 mão esquerda, 112-132 mão direita.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from settings import get_settings

N_KPTS = 133
FACE = slice(23, 91)
# boca no rosto de 68 pontos (48-67) -> 23+48 .. 23+67
MOUTH = slice(23 + 48, 23 + 68)


@dataclass
class PoseEstimator:
    onnx: str = field(default_factory=lambda: get_settings().dwpose_pose_onnx)
    device: str = field(default_factory=lambda: get_settings().device)
    input_size: tuple[int, int] = (288, 384)
    _model: object = None

    def load(self):
        if self._model is None:
            from rtmlib import RTMPose
            self._model = RTMPose(self.onnx, model_input_size=self.input_size,
                                  backend="onnxruntime", device=self.device)
        return self._model

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """-> (N, 133, 3) com x, y, score em pixels da imagem."""
        if len(boxes) == 0:
            return np.zeros((0, N_KPTS, 3), np.float32)
        kpts, scores = self.load()(image, bboxes=boxes.tolist())
        return np.concatenate([kpts, scores[..., None]], axis=-1).astype(np.float32)


def face_box(kpts: np.ndarray, min_score: float = 0.3, pad: float = 0.25) -> np.ndarray | None:
    """Caixa do rosto a partir dos 68 pontos faciais (para o lipsync)."""
    f = kpts[FACE]
    f = f[f[:, 2] >= min_score]
    if len(f) < 10:
        return None
    x1, y1 = f[:, 0].min(), f[:, 1].min()
    x2, y2 = f[:, 0].max(), f[:, 1].max()
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h], np.float32)
