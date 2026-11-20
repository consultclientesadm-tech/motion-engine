"""Inpainting sem GPU: segmentador falso (máscara = caixa) e reserva do cv2.inpaint."""
import shutil
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="sem ffmpeg")

from workers.analyze.blockdata import BlockAnalysis  # noqa: E402
from workers.inpaint import clean  # noqa: E402

W, H, N = 160, 90, 30


class BoxSeg:
    def masks(self, image, targets):
        return clean.box_mask(targets, *image.shape[:2], pad=0.0)


class NoLama(clean.Lama):
    def __init__(self):
        super().__init__(weights="/nao/existe.pt", device="cpu")


def _background():
    y, x = np.mgrid[0:H, 0:W]
    return np.dstack([(x * 255 // W), (y * 255 // H), np.full((H, W), 90)]).astype(np.uint8)


def _clip(tmp_path: Path):
    """Fundo fixo; um 'pessoa' (quadrado magenta) anda da esquerda para a direita."""
    bg = _background()
    boxes = []
    path = tmp_path / "src.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i in range(N):
        img = bg.copy()
        x1 = 10 + i * 4
        img[30:70, x1:x1 + 20] = (255, 0, 255)
        boxes.append((x1, 30, x1 + 20, 70))
        out.write(img)
    out.release()
    return path, bg, boxes


def test_temporal_background_ignores_masked_pixels():
    bg = _background()
    frames, masks = [], []
    for i in range(6):
        f = bg.copy(); m = np.zeros((H, W), np.uint8)
        f[10:40, i * 20:i * 20 + 20] = 255; m[10:40, i * 20:i * 20 + 20] = 255
        frames.append(f); masks.append(m)
    got, hole = clean.temporal_background(frames, masks)
    assert not hole.any()
    assert np.abs(got.astype(int) - bg).max() <= 1


def test_clean_block_removes_mapped_person_only(tmp_path):
    src, bg, boxes = _clip(tmp_path)
    rows = [(i, 7, np.array(b, np.float32), np.zeros((133, 3), np.float32)) for i, b in enumerate(boxes)]
    an = BlockAnalysis.from_rows(rows, 0, N)
    out = tmp_path / "clean.mp4"

    st = clean.clean_block(source_block=src, analysis=an, keep_tracks={7}, src_w=W, src_h=H, out=out,
                           segmenter=BoxSeg(), lama=NoLama(), sample_every=3)
    assert st.method == "temporal" and st.masked_ratio > 0
    cap = cv2.VideoCapture(str(out)); ok, f = cap.read(); cap.release()
    assert ok and f.shape == (H, W, 3)
    x1, y1, x2, y2 = boxes[0]
    region = f[y1 + 4:y2 - 4, x1 + 4:x2 - 4].astype(int)
    # magenta (255,0,255) sumiu: ficou perto do fundo de verdade
    assert np.abs(region - bg[y1 + 4:y2 - 4, x1 + 4:x2 - 4]).mean() < 25

    # track que não vira personagem fica no quadro
    st2 = clean.clean_block(source_block=src, analysis=an, keep_tracks={99}, src_w=W, src_h=H,
                            out=tmp_path / "keep.mp4", segmenter=BoxSeg(), lama=NoLama())
    assert st2.method == "none"


def test_lama_fallback_fills_hole():
    img = _background(); m = np.zeros((H, W), np.uint8); m[40:50, 70:80] = 255
    img[40:50, 70:80] = 0
    out = NoLama()(img, m)
    assert out[45, 75].sum() > 0
