"""Roda DENTRO do Blender: blender -b --python blender_render.py -- cfg.json

Renderiza um bloco: carrega os avatares (.blend), aplica a pose de cada frame
(vinda de workers/retarget) e grava PNG RGBA com fundo transparente. A
composição sobre o vídeo original é feita depois pelo FFmpeg.
"""
import json
import math
import sys

import bpy

cfg = json.load(open(sys.argv[sys.argv.index("--") + 1]))
W, H, FPS = cfg["width"], cfg["height"], cfg["fps"]
AV_H = cfg["avatar_height"]
HIP_Z = 0.53 * AV_H
FRAME_H = 10.0                    # altura do quadro em unidades do mundo
FRAME_W = FRAME_H * W / H

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
scene.render.fps = int(round(FPS))
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
for engine in (cfg.get("engine"), "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    if not engine:
        continue
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue
if scene.render.engine == "CYCLES":
    scene.cycles.samples = 16
    scene.cycles.device = "GPU"
else:
    try:
        scene.eevee.taa_render_samples = 16
    except AttributeError:
        pass

# câmera ortográfica olhando +Y: (u, v) normalizado -> (x, z) do mundo
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = max(FRAME_W, FRAME_H)
cam = bpy.data.objects.new("cam", cam_data)
cam.location = (0, -50, 0)
cam.rotation_euler = (math.radians(90), 0, 0)
scene.collection.objects.link(cam)
scene.camera = cam

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.rotation_euler = (math.radians(60), 0, math.radians(-20))
sun.data.energy = 3.0
scene.collection.objects.link(sun)
world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
scene.world = world


def uv_to_world(u, v):
    return (u - 0.5) * FRAME_W, (0.5 - v) * FRAME_H


# ---- avatares ------------------------------------------------------------------
rigs = {}
for char, blend in cfg["avatars"].items():
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in ("avatar_mesh", "avatar_rig")]
    rig = mesh = None
    for ob in dst.objects:
        scene.collection.objects.link(ob)
        ob.name = f"{char}_{ob.type.lower()}"
        if ob.type == "ARMATURE":
            rig = ob
        else:
            mesh = ob
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    rigs[char] = (rig, mesh)

bones = cfg["bones"]          # {osso: {"parent": pai|None}}
rest = cfg["rest_angles"]


def local(world_angles):
    out = {}
    for b, meta in bones.items():
        p = meta["parent"]
        wp = world_angles[p] if p else 0.0
        rp = rest[p] if p else 0.0
        out[b] = (world_angles[b] - wp) - (rest[b] - rp)
    return out


frames = cfg["frames"]
scene.frame_start, scene.frame_end = 1, len(frames)
for fr in frames:
    f = fr["f"] + 1
    present = {a["character"]: a for a in fr["actors"]}
    for char, (rig, mesh) in rigs.items():
        a = present.get(char)
        hidden = a is None
        for ob in (rig, mesh):
            ob.hide_render = hidden
            ob.keyframe_insert("hide_render", frame=f)
        if hidden:
            continue
        s = a["scale"] * FRAME_H / AV_H
        x, z = uv_to_world(*a["root"])
        rig.scale = (s, s, s)
        rig.location = (x, 0, z - HIP_Z * s)
        rig.keyframe_insert("scale", frame=f)
        rig.keyframe_insert("location", frame=f)
        for b, ang in local(a["angles"]).items():
            pb = rig.pose.bones.get(b)
            if pb is None:
                continue
            pb.rotation_euler = (0, 0, math.radians(ang))
            pb.keyframe_insert("rotation_euler", frame=f)

# poses discretas por frame: sem interpolação entre keyframes
for rig, mesh in rigs.values():
    for ob in (rig, mesh):
        if ob.animation_data and ob.animation_data.action:
            try:
                for fc in ob.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"
            except AttributeError:  # Blender 5 (actions em camadas): mantém padrão
                pass

scene.render.filepath = cfg["out_dir"].rstrip("/") + "/frame_"
bpy.ops.render.render(animation=True)
print("[render] ok", len(frames), "frames")
