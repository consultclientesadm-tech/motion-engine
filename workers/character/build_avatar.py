"""Avatar aproximado a partir da folha de referência.

1. Recorta a vista frontal da folha (folhas costumam ter frente/lado/costas
   lado a lado; por padrão usa o painel mais à esquerda).
2. Remove o fundo (rembg) e gera a malha com TripoSR (cores por vértice).
3. Blender headless: normaliza (em pé, 1,75 m, pés no chão, encarando -Y),
   cria o esqueleto de workers/retarget e faz o skin automático. Exporta .glb.

InstantMesh pode substituir o TripoSR implementando `mesh_from_image` com a
mesma assinatura (AVATAR_BACKEND=instantmesh).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from settings import get_settings
from workers.media import run

RIG_SCRIPT = Path(__file__).with_name("blender_rig.py")


def front_view(sheet_path: Path, out: Path, panels: int | None = None) -> Path:
    from PIL import Image
    img = Image.open(sheet_path).convert("RGBA")
    w, h = img.size
    if panels is None:  # heurística: folha larga = vários painéis de ~ h/2 de largura
        panels = max(1, round(w / (h * 0.55))) if w > h * 1.2 else 1
    if panels > 1:
        img = img.crop((0, 0, w // panels, h))
    img.save(out)
    return out


def mesh_from_image(image_path: Path, out_obj: Path, resolution: int = 256) -> Path:
    import numpy as np
    import rembg
    import torch
    from PIL import Image
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    s = get_settings()
    device = s.device if torch.cuda.is_available() else "cpu"
    model = TSR.from_pretrained(s.triposr_model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)

    image = remove_background(Image.open(image_path), rembg.new_session())
    image = resize_foreground(image, 0.85)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    image = Image.fromarray((arr * 255.0).astype(np.uint8))

    with torch.no_grad():
        codes = model([image], device=device)
        mesh = model.extract_mesh(codes, True, resolution=resolution)[0]
    mesh.export(str(out_obj))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_obj


def rig_mesh(obj_path: Path, out_glb: Path) -> Path:
    s = get_settings()
    from workers.retarget.retarget import AVATAR_HEIGHT, BONES, REST_ANGLES
    cfg = out_glb.with_suffix(".rig.json")
    cfg.write_text(json.dumps({"mesh": str(obj_path), "out": str(out_glb), "height": AVATAR_HEIGHT,
                               "bones": {b: p for b, (_, _, p) in BONES.items()},
                               "rest_angles": REST_ANGLES}))
    run([s.blender_bin, "-b", "--factory-startup", "--python", str(RIG_SCRIPT), "--", str(cfg)],
        timeout=900)
    if not out_glb.exists():
        raise RuntimeError("Blender não gerou o .glb do avatar")
    return out_glb


def build_avatar(sheet_path: Path, workdir: Path) -> Path:
    """Folha -> avatar .glb com rig. Retorna o caminho local do .glb."""
    front = front_view(sheet_path, workdir / "front.png",
                       panels=int(os.environ["SHEET_PANELS"]) if os.environ.get("SHEET_PANELS") else None)
    obj = mesh_from_image(front, workdir / "mesh.obj")
    return rig_mesh(obj, workdir / "avatar.glb")
