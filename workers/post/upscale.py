"""Upscale 720p -> 1080p com Real-ESRGAN (x2plus, saída redimensionada x1.5).

Streaming: lê frame a frame com OpenCV e escreve num pipe do FFmpeg, então o
bloco de 30 s nunca fica inteiro na memória.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from settings import get_settings
from workers.media import ffmpeg


class Upscaler:
    def __init__(self, tile: int = 0):
        self.tile = tile
        self._model = None

    def load(self):
        if self._model is None:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            s = get_settings()
            gpu = torch.cuda.is_available()
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self._model = RealESRGANer(scale=2, model_path=s.realesrgan_weights, model=net,
                                       tile=self.tile, tile_pad=10, pre_pad=0, half=gpu,
                                       device="cuda" if gpu else "cpu")
        return self._model

    def upscale_video(self, src: Path, dst: Path, fps: float) -> Path:
        import cv2
        s = get_settings()
        model = self.load()
        W, H = s.output_width, s.output_height
        cap = cv2.VideoCapture(str(src))
        enc = subprocess.Popen([s.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                                "-video_track_timescale", "90000", str(dst)], stdin=subprocess.PIPE)
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                outscale = W / img.shape[1]  # 1280 -> 1920 = 1.5
                up, _ = model.enhance(img, outscale=outscale)
                if up.shape[1] != W or up.shape[0] != H:
                    up = cv2.resize(up, (W, H), interpolation=cv2.INTER_LANCZOS4)
                enc.stdin.write(up.tobytes())
        finally:
            cap.release()
            enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg falhou ao gravar o bloco em 1080p")
        return dst


def upscale(src: Path, dst: Path, fps: float, upscaler: Upscaler | None = None) -> Path:
    return (upscaler or Upscaler()).upscale_video(src, dst, fps)


def passthrough(src: Path, dst: Path, start_s: float, end_s: float, width: int, height: int,
                fps: float) -> Path:
    """Bloco sem pessoa: recorta o original direto na resolução final (lanczos), sem GPU."""
    ffmpeg("-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{end_s - start_s:.3f}",
           "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos", "-video_track_timescale", "90000",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dst))
    return dst
