"""Roda DENTRO do Blender: blender -b --python blender_rig.py -- cfg.json

Normaliza a malha do TripoSR, cria o esqueleto (A-pose, ossos girando no
plano XZ) e faz o skin. Salva avatar.blend (usado pelo render) e avatar.glb
(prévia). Variável AVATAR_EULER="x,y,z" (graus) corrige a orientação da malha
se o seu TripoSR exportar deitado ou de costas.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

cfg = json.load(open(sys.argv[sys.argv.index("--") + 1]))
H = cfg["height"]

bpy.ops.wm.read_factory_settings(use_empty=True)

# ---- malha -----------------------------------------------------------------
bpy.ops.wm.obj_import(filepath=cfg["mesh"])
mesh = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
mesh.name = "avatar_mesh"
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)

euler = os.environ.get("AVATAR_EULER")
if euler:
    mesh.rotation_euler = Euler([math.radians(float(v)) for v in euler.split(",")])
else:
    # eixo mais longo vira Z (em pé)
    dims = mesh.dimensions
    longest = max(range(3), key=lambda i: dims[i])
    if longest == 0:
        mesh.rotation_euler = Euler((0, math.radians(90), 0))
    elif longest == 1:
        mesh.rotation_euler = Euler((math.radians(90), 0, 0))
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# escala para H e pés em z=0, centrado em x/y
bbox = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
zmin, zmax = min(v.z for v in bbox), max(v.z for v in bbox)
k = H / (zmax - zmin)
mesh.scale = (k, k, k)
bpy.ops.object.transform_apply(scale=True)
bbox = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
cx = (min(v.x for v in bbox) + max(v.x for v in bbox)) / 2
cy = (min(v.y for v in bbox) + max(v.y for v in bbox)) / 2
mesh.location -= Vector((cx, cy, min(v.z for v in bbox)))
bpy.ops.object.transform_apply(location=True)

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.0005)
bpy.ops.object.mode_set(mode="OBJECT")

# material com as cores por vértice do TripoSR
mat = bpy.data.materials.new("avatar_mat")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if mesh.data.color_attributes:
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = mesh.data.color_attributes[0].name
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.8
mesh.data.materials.clear()
mesh.data.materials.append(mat)

# ---- esqueleto ---------------------------------------------------------------
def d(angle_deg):
    a = math.radians(angle_deg)
    return Vector((math.sin(a), 0, math.cos(a)))

hip = Vector((0, 0, 0.53 * H))
sh = Vector((0, 0, 0.83 * H))
joints = {
    "root": (hip, hip + Vector((0, 0, 0.04 * H))),
    "spine": (hip, sh),
    "head": (sh, sh + Vector((0, 0, 0.10 * H))),
}
for side, sx in (("L", 1), ("R", -1)):
    rest = cfg["rest_angles"]
    s0 = Vector((sx * 0.11 * H, 0, 0.82 * H))
    e = s0 + d(rest[f"upper_arm.{side}"]) * 0.17 * H
    joints[f"upper_arm.{side}"] = (s0, e)
    joints[f"forearm.{side}"] = (e, e + d(rest[f"forearm.{side}"]) * 0.15 * H)
    h0 = Vector((sx * 0.055 * H, 0, 0.52 * H))
    kn = h0 + d(rest[f"thigh.{side}"]) * 0.245 * H
    joints[f"thigh.{side}"] = (h0, kn)
    joints[f"shin.{side}"] = (kn, kn + d(rest[f"shin.{side}"]) * 0.25 * H)

arm_data = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("avatar_rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for name, (h, t) in joints.items():
    b = arm_data.edit_bones.new(name)
    b.head, b.tail = h, t
    b.align_roll(Vector((0, 1, 0)))   # eixo Z local = Y do mundo (eixo da câmera)
    eb[name] = b
parents = dict(cfg["bones"])
for name, b in eb.items():
    if name == "root":
        continue
    p = parents.get(name) or "root"
    b.parent = eb[p]
    b.use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")
for pb in arm.pose.bones:
    pb.rotation_mode = "XYZ"

# ---- skin ----------------------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
unweighted = sum(1 for v in mesh.data.vertices if not v.groups)
if unweighted > 0.05 * len(mesh.data.vertices):
    print(f"[rig] heat weighting falhou em {unweighted} vértices; usando envelope")
    mesh.vertex_groups.clear()
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")

out = cfg["out"]
bpy.ops.wm.save_as_mainfile(filepath=out.replace(".glb", ".blend"))
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
print("[rig] ok", out)
