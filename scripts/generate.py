"""Generate synthetic Mikado training images using Blender.

Run headless:
    blender --background --python scripts/generate.py -- \
        --count 200 --output output/ --config configs/

Each render:
  1. Clears the scene
  2. Creates a table plane
  3. Spawns sticks as cylinders with coloured tips
  4. Runs rigid-body physics so sticks fall into a natural pile
  5. Renders from above (orthographic-ish perspective)
  6. Exports YOLO-OBB label file alongside each image

Label format (matches mikado-judge):
    class_id x1 y1 x2 y2 x3 y3 x4 y4   (normalised 0-1)
"""

import bpy
import bmesh
import math
import os
import random
import sys
import yaml
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# Argument parsing — Blender passes script args after '--'
# ---------------------------------------------------------------------------
argv = sys.argv
script_args = argv[argv.index("--") + 1:] if "--" in argv else []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=50, help="Number of images to generate")
parser.add_argument("--output", default="output", help="Output directory")
parser.add_argument("--config", default="configs", help="Config directory")
parser.add_argument("--seed", type=int, default=0, help="Random seed (0 = random)")
parser.add_argument("--stick-style", choices=["simple", "realistic"], default="simple",
                    help="Stick material style: simple (flat color) or realistic (wood grain + ring tips)")
parser.add_argument("--resolution", choices=["default", "highres"], default="default",
                    help="Output resolution: default (1280x960) or highres (4624x3472 for tiling)")
args = parser.parse_args(script_args)

random.seed(args.seed if args.seed != 0 else None)
os.makedirs(args.output, exist_ok=True)
labels_dir = os.path.join(args.output, "labels")
images_dir = os.path.join(args.output, "images")
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------
with open(os.path.join(args.config, "sticks.yaml")) as f:
    STICKS = yaml.safe_load(f)
with open(os.path.join(args.config, "render.yaml")) as f:
    RENDER = yaml.safe_load(f)

NAMES = STICKS["names"]           # {0: "mikado", 1: "blue", ...}
NAME_TO_ID = {v: k for k, v in NAMES.items()}
COUNTS = STICKS["counts"]
TIP_COLORS = STICKS["tip_colors"]
BODY_COLOR = STICKS["body_color"]
TIP_FRAC = STICKS["tip_fraction"]
L = STICKS["dimensions"]["length_mm"] / 1000.0   # metres (Blender default unit)
D = STICKS["dimensions"]["diameter_mm"] / 1000.0

_output_key = "output_highres" if args.resolution == "highres" else "output"
_output_cfg = RENDER.get(_output_key, RENDER["output"])
W_OUT = _output_cfg["width"]
H_OUT = _output_cfg["height"]
SAMPLES = _output_cfg["samples"]
ENGINE = _output_cfg["engine"]

CAM_H = RENDER["camera"]["height_mm"] / 1000.0

import math as _math
# Compute OBB thickness in pixels from actual camera geometry so boxes
# match the rendered stick width exactly (like real annotated data).
_fov = _math.radians(RENDER["camera"]["fov_deg"])
_coverage_w = 2 * CAM_H * _math.tan(_fov / 2)        # metres visible at table level
THICKNESS_PX = max(4, round(D / _coverage_w * W_OUT))
del _math
DROP_W = RENDER["drop_zone"]["width_mm"] / 1000.0
DROP_H = RENDER["drop_zone"]["height_mm"] / 1000.0
DROP_HEIGHT = RENDER["physics"]["drop_height_mm"] / 1000.0
SIM_STEPS = RENDER["physics"]["sim_steps"]


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def clear_scene():
    # Free physics cache before removing the world so Bullet releases its
    # internal state cleanly — prevents stale references on scene 2+.
    if bpy.context.scene.rigidbody_world is not None:
        try:
            pc = bpy.context.scene.rigidbody_world.point_cache
            bpy.ops.ptcache.free_bake({"point_cache": pc})
        except Exception:
            pass
        bpy.ops.rigidbody.world_remove()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)


def make_material(name, color, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_cloth_material(name, color, roughness=0.85):
    """Procedural cloth/felt material with subtle wrinkles and color variation."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-800, 0)

    # --- Color variation: low-frequency noise shifts the base color slightly ---
    noise_color = nodes.new("ShaderNodeTexNoise")
    noise_color.location = (-400, 200)
    noise_color.inputs["Scale"].default_value = random.uniform(5.0, 15.0)
    noise_color.inputs["Detail"].default_value = 3.0
    noise_color.inputs["Roughness"].default_value = 0.6
    links.new(tex_coord.outputs["Object"], noise_color.inputs["Vector"])

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-200, 200)
    # Base color at 0.4, slightly darker variant at 0.6
    r, g, b = color
    darken = 0.85
    color_ramp.color_ramp.elements[0].position = 0.4
    color_ramp.color_ramp.elements[0].color = (r, g, b, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.6
    color_ramp.color_ramp.elements[1].color = (r * darken, g * darken, b * darken, 1.0)
    links.new(noise_color.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # --- Bump: high-frequency noise for fine fabric texture ---
    noise_fine = nodes.new("ShaderNodeTexNoise")
    noise_fine.location = (-400, -100)
    noise_fine.inputs["Scale"].default_value = random.uniform(80.0, 150.0)
    noise_fine.inputs["Detail"].default_value = 6.0
    noise_fine.inputs["Roughness"].default_value = 0.7
    links.new(tex_coord.outputs["Object"], noise_fine.inputs["Vector"])

    # Voronoi for weave-like grain
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.location = (-400, -300)
    voronoi.inputs["Scale"].default_value = random.uniform(200.0, 400.0)
    links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])

    # Mix noise + voronoi for combined bump
    mix_bump = nodes.new("ShaderNodeMixRGB")
    mix_bump.location = (-200, -200)
    mix_bump.inputs["Fac"].default_value = 0.3  # mostly noise, some voronoi grain
    links.new(noise_fine.outputs["Fac"], mix_bump.inputs["Color1"])
    links.new(voronoi.outputs["Distance"], mix_bump.inputs["Color2"])

    # Large-scale wrinkles (low-frequency noise for cloth folds)
    noise_wrinkle = nodes.new("ShaderNodeTexNoise")
    noise_wrinkle.location = (-400, -500)
    noise_wrinkle.inputs["Scale"].default_value = random.uniform(2.0, 6.0)
    noise_wrinkle.inputs["Detail"].default_value = 4.0
    noise_wrinkle.inputs["Roughness"].default_value = 0.5
    links.new(tex_coord.outputs["Object"], noise_wrinkle.inputs["Vector"])

    # Combine fine texture + wrinkles
    mix_all = nodes.new("ShaderNodeMixRGB")
    mix_all.location = (-50, -300)
    mix_all.inputs["Fac"].default_value = 0.4
    links.new(mix_bump.outputs["Color"], mix_all.inputs["Color1"])
    links.new(noise_wrinkle.outputs["Fac"], mix_all.inputs["Color2"])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (100, -200)
    bump.inputs["Strength"].default_value = random.uniform(0.02, 0.05)
    bump.inputs["Distance"].default_value = 0.1
    links.new(mix_all.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def add_table(color):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    table = bpy.context.active_object
    table.name = "Table"
    mat = make_cloth_material("TableMat", color, roughness=random.uniform(0.8, 0.95))
    table.data.materials.append(mat)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.rigidbody.object_add()
    table.rigid_body.type = "PASSIVE"
    table.rigid_body.collision_shape = "MESH"
    table.rigid_body.use_margin = True
    table.rigid_body.collision_margin = 0.001  # 1mm — default 40mm causes hovering
    return table


def make_stick_material(name, wood_color, tip_color):
    """Unified stick material: smooth bamboo body with sharp dipped-paint tips.

    Uses Object-space X position to blend wood color into tip paint color
    with a sharp boundary (like real dipped paint). Compatible with Blender 2.82+.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    for n in nodes:
        nodes.remove(n)

    tip_len = L * TIP_FRAC
    fade_start = L / 2 - tip_len          # paint starts exactly at tip boundary
    fade_range = tip_len * 0.15           # sharp transition (~3mm)

    # --- Output + BSDF ---
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (800, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (550, 0)
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.6
            break
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # --- Texture coordinates ---
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1200, 0)

    # === WOOD BASE COLOR (subtle noise, no grain bands) ===
    noise_wood = nodes.new("ShaderNodeTexNoise")
    noise_wood.location = (-800, 300)
    noise_wood.inputs["Scale"].default_value = random.uniform(200.0, 400.0)
    noise_wood.inputs["Detail"].default_value = 4.0
    noise_wood.inputs["Roughness"].default_value = 0.6
    links.new(tex_coord.outputs["Object"], noise_wood.inputs["Vector"])

    wood_ramp = nodes.new("ShaderNodeValToRGB")
    wood_ramp.location = (-500, 300)
    wr, wg, wb = wood_color
    wood_ramp.color_ramp.elements[0].position = 0.45
    wood_ramp.color_ramp.elements[0].color = (
        min(1, wr * 1.04), min(1, wg * 1.04), min(1, wb * 1.04), 1.0)
    wood_ramp.color_ramp.elements[1].position = 0.55
    wood_ramp.color_ramp.elements[1].color = (wr * 0.96, wg * 0.96, wb * 0.96, 1.0)
    links.new(noise_wood.outputs["Fac"], wood_ramp.inputs["Fac"])

    # === FADE GRADIENT along X axis ===
    # TexCoord → SeparateXYZ → |X| → subtract fade_start → divide by fade_range → clamp
    sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
    sep_xyz.location = (-900, -100)
    links.new(tex_coord.outputs["Object"], sep_xyz.inputs["Vector"])

    abs_x = nodes.new("ShaderNodeMath")
    abs_x.location = (-700, -100)
    abs_x.operation = "ABSOLUTE"
    links.new(sep_xyz.outputs["X"], abs_x.inputs[0])

    sub_start = nodes.new("ShaderNodeMath")
    sub_start.location = (-500, -100)
    sub_start.operation = "SUBTRACT"
    sub_start.inputs[1].default_value = fade_start
    links.new(abs_x.outputs["Value"], sub_start.inputs[0])

    div_range = nodes.new("ShaderNodeMath")
    div_range.location = (-300, -100)
    div_range.operation = "DIVIDE"
    div_range.inputs[1].default_value = fade_range
    links.new(sub_start.outputs["Value"], div_range.inputs[0])

    clamp_fade = nodes.new("ShaderNodeMath")
    clamp_fade.location = (-100, -100)
    clamp_fade.operation = "MINIMUM"
    clamp_fade.inputs[1].default_value = 1.0
    links.new(div_range.outputs["Value"], clamp_fade.inputs[0])

    clamp_fade2 = nodes.new("ShaderNodeMath")
    clamp_fade2.location = (50, -100)
    clamp_fade2.operation = "MAXIMUM"
    clamp_fade2.inputs[1].default_value = 0.0
    links.new(clamp_fade.outputs["Value"], clamp_fade2.inputs[0])

    # Noise perturbation on fade boundary
    noise_fade = nodes.new("ShaderNodeTexNoise")
    noise_fade.location = (-300, -300)
    noise_fade.inputs["Scale"].default_value = random.uniform(400.0, 600.0)
    noise_fade.inputs["Detail"].default_value = 3.0
    noise_fade.inputs["Roughness"].default_value = 0.5
    links.new(tex_coord.outputs["Object"], noise_fade.inputs["Vector"])

    # Scale noise to small perturbation and center around 0
    noise_sub = nodes.new("ShaderNodeMath")
    noise_sub.location = (-100, -300)
    noise_sub.operation = "SUBTRACT"
    noise_sub.inputs[1].default_value = 0.5
    links.new(noise_fade.outputs["Fac"], noise_sub.inputs[0])

    noise_mul = nodes.new("ShaderNodeMath")
    noise_mul.location = (50, -300)
    noise_mul.operation = "MULTIPLY"
    noise_mul.inputs[1].default_value = 0.08  # ±0.04 — subtle edge irregularity
    links.new(noise_sub.outputs["Value"], noise_mul.inputs[0])

    # Add noise to fade gradient
    fade_noisy = nodes.new("ShaderNodeMath")
    fade_noisy.location = (200, -200)
    fade_noisy.operation = "ADD"
    links.new(clamp_fade2.outputs["Value"], fade_noisy.inputs[0])
    links.new(noise_mul.outputs["Value"], fade_noisy.inputs[1])

    # Final clamp
    fade_final = nodes.new("ShaderNodeMath")
    fade_final.location = (350, -200)
    fade_final.operation = "MINIMUM"
    fade_final.inputs[1].default_value = 1.0
    links.new(fade_noisy.outputs["Value"], fade_final.inputs[0])

    fade_final2 = nodes.new("ShaderNodeMath")
    fade_final2.location = (350, -350)
    fade_final2.operation = "MAXIMUM"
    fade_final2.inputs[1].default_value = 0.0
    links.new(fade_final.outputs["Value"], fade_final2.inputs[0])

    # === PRE-BLEND TIP COLOR with wood for soaked-in look (85% paint, 15% wood) ===
    tr, tg, tb = tip_color
    blended_tip = nodes.new("ShaderNodeMixRGB")
    blended_tip.location = (-200, 150)
    blended_tip.inputs["Fac"].default_value = 0.95
    blended_tip.inputs["Color1"].default_value = (wr, wg, wb, 1.0)
    blended_tip.inputs["Color2"].default_value = (tr, tg, tb, 1.0)

    # === MIX wood ↔ tip using fade factor ===
    mix_final = nodes.new("ShaderNodeMixRGB")
    mix_final.location = (350, 100)
    links.new(fade_final2.outputs["Value"], mix_final.inputs["Fac"])
    links.new(wood_ramp.outputs["Color"], mix_final.inputs["Color1"])
    links.new(blended_tip.outputs["Color"], mix_final.inputs["Color2"])
    links.new(mix_final.outputs["Color"], bsdf.inputs["Base Color"])

    # === ROUGHNESS GRADIENT: wood ~0.5, paint ~0.3 ===
    rough_mix = nodes.new("ShaderNodeMath")
    rough_mix.location = (350, -500)
    rough_mix.operation = "MULTIPLY"
    rough_mix.inputs[1].default_value = -0.2  # roughness delta
    links.new(fade_final2.outputs["Value"], rough_mix.inputs[0])

    rough_add = nodes.new("ShaderNodeMath")
    rough_add.location = (500, -500)
    rough_add.operation = "ADD"
    rough_add.inputs[1].default_value = random.uniform(0.45, 0.55)  # wood roughness
    links.new(rough_mix.outputs["Value"], rough_add.inputs[0])
    links.new(rough_add.outputs["Value"], bsdf.inputs["Roughness"])

    # === SUBTLE BUMP from noise ===
    bump = nodes.new("ShaderNodeBump")
    bump.location = (350, -650)
    bump.inputs["Strength"].default_value = random.uniform(0.01, 0.02)
    bump.inputs["Distance"].default_value = 0.05
    links.new(noise_wood.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def add_stick_realistic(class_name, location, rotation_euler):
    """Create a stick with smooth bamboo body, tapered ends, and fading paint tips.

    Single mesh with vertex-based taper (narrowing toward both ends) and a
    unified shader that blends wood color into soaked-in paint at the tips.
    """
    circ_segs = 12   # vertices around circumference
    length_rings = 20  # rings along the stick length (enough for smooth taper)
    TAPER_START_FRAC = 0.75  # taper begins at 75% of half-length from center
    TIP_DIAM_FRAC = 0.55     # tip narrows to 55% of full diameter

    mesh = bpy.data.meshes.new(f"stick_mesh_{class_name}")
    obj = bpy.data.objects.new(f"stick_{class_name}_{random.randint(0,99999)}", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()

    # Build stick as a series of rings along local X with per-ring radius
    half_L = L / 2
    R = D / 2

    # Create ring vertices
    rings = []
    for i in range(length_rings + 1):
        x = -half_L + i * L / length_rings
        t = abs(x) / half_L  # 0 at center, 1 at tip
        if t > TAPER_START_FRAC:
            blend = (t - TAPER_START_FRAC) / (1.0 - TAPER_START_FRAC)
            blend = 1.0 - math.cos(blend * math.pi / 2)  # cosine ease-in
            radius = R * (1.0 + (TIP_DIAM_FRAC - 1.0) * blend)
        else:
            radius = R
        ring_verts = []
        for j in range(circ_segs):
            angle = 2 * math.pi * j / circ_segs
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)
            ring_verts.append(bm.verts.new((x, y, z)))
        rings.append(ring_verts)

    bm.verts.ensure_lookup_table()

    # Connect adjacent rings with quad faces
    for i in range(len(rings) - 1):
        for j in range(circ_segs):
            j_next = (j + 1) % circ_segs
            bm.faces.new([
                rings[i][j], rings[i][j_next],
                rings[i + 1][j_next], rings[i + 1][j],
            ])

    # Cap ends
    for ring in (rings[0], rings[-1]):
        center = bm.verts.new((ring[0].co.x, 0, 0))
        for j in range(circ_segs):
            j_next = (j + 1) % circ_segs
            if ring is rings[0]:
                bm.faces.new([center, ring[j_next], ring[j]])
            else:
                bm.faces.new([center, ring[j], ring[j_next]])

    bm.normal_update()

    bm.to_mesh(mesh)
    bm.free()

    # Unified material: wood body + fading paint tips
    r, g, b = BODY_COLOR
    jitter = 0.06
    wood_color = (
        max(0, min(1, r + random.uniform(-jitter, jitter))),
        max(0, min(1, g + random.uniform(-jitter, jitter))),
        max(0, min(1, b + random.uniform(-jitter, jitter))),
    )
    obj.data.materials.append(make_stick_material(
        f"mat_{obj.name}", wood_color, TIP_COLORS[class_name]))

    obj.location = location
    obj.rotation_euler = rotation_euler

    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "ACTIVE"
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.001
    obj.rigid_body.mass = 0.005
    obj.rigid_body.restitution = 0.05
    obj.rigid_body.friction = 0.6
    obj.rigid_body.angular_damping = 0.85
    obj.rigid_body.linear_damping = 0.85

    obj["class_name"] = class_name
    return obj


def add_stick(class_name, location, rotation_euler):
    """Create a stick with coloured tips, built horizontally in local space via bmesh.

    The long axis is along local X so CONVEX_HULL is computed from a horizontal
    cylinder. No primitive_cylinder_add + join — avoids the join-resets-rotation
    bug that caused sticks to hover.
    """
    tip_len = L * TIP_FRAC
    segs = 8

    mesh = bpy.data.meshes.new(f"stick_mesh_{class_name}")
    obj = bpy.data.objects.new(f"stick_{class_name}_{random.randint(0,99999)}", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()

    # Body — cylinder along local X
    bmesh.ops.create_cone(bm,
        cap_ends=True, cap_tris=False, segments=segs,
        radius1=D / 2, radius2=D / 2, depth=L,
        matrix=Matrix.Rotation(math.pi / 2, 4, 'Y'),
    )
    for f in bm.faces:
        f.material_index = 0

    # Tip caps at each end along X
    for sign in (+1, -1):
        tip_matrix = (Matrix.Translation((sign * (L / 2 - tip_len / 2), 0, 0))
                      @ Matrix.Rotation(math.pi / 2, 4, 'Y'))
        bmesh.ops.create_cone(bm,
            cap_ends=True, cap_tris=False, segments=segs,
            radius1=D / 2 + 0.0001, radius2=D / 2 + 0.0001, depth=tip_len,
            matrix=tip_matrix,
        )

    # Assign tip material to faces whose centroid is near the ends
    for f in bm.faces:
        cx = sum(v.co.x for v in f.verts) / len(f.verts)
        f.material_index = 1 if abs(cx) > (L / 2 - tip_len - 0.001) else 0

    bm.to_mesh(mesh)
    bm.free()

    r, g, b = BODY_COLOR
    jitter = 0.06
    wood_color = (
        max(0, min(1, r + random.uniform(-jitter, jitter))),
        max(0, min(1, g + random.uniform(-jitter, jitter))),
        max(0, min(1, b + random.uniform(-jitter, jitter))),
    )
    obj.data.materials.append(make_material(f"body_{obj.name}", wood_color, roughness=random.uniform(0.5, 0.8)))
    obj.data.materials.append(make_material(f"tip_{obj.name}", TIP_COLORS[class_name], roughness=0.3))

    obj.location = location
    obj.rotation_euler = rotation_euler

    # CONVEX_HULL uses local verts (horizontal) × matrix_world (spawn rotation).
    # No transform_apply needed — mesh is already in the correct local orientation.
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "ACTIVE"
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.001  # 1mm — default 40mm >> 3mm stick diameter
    obj.rigid_body.mass = 0.005
    obj.rigid_body.restitution = 0.05
    obj.rigid_body.friction = 0.6
    obj.rigid_body.angular_damping = 0.85   # high damping so sticks settle into a tight pile
    obj.rigid_body.linear_damping = 0.85   # high damping so sticks stop sliding

    obj["class_name"] = class_name
    return obj


def setup_camera():
    bpy.ops.object.camera_add(location=(0, 0, CAM_H))
    cam = bpy.context.active_object
    cam.name = "Camera"

    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.active_object
    target.name = "CameraTarget"

    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    bpy.context.scene.camera = cam

    fov = math.radians(RENDER["camera"]["fov_deg"])
    cam.data.lens_unit = "FOV"
    cam.data.angle = fov

    half_w = CAM_H * math.tan(fov / 2)
    print(f'Camera at Z={CAM_H:.3f}m, FOV={RENDER["camera"]["fov_deg"]}°, '
          f'covers ±{half_w*1000:.0f}mm at table level')

    return cam


def setup_lighting():
    bpy.ops.object.light_add(type="SUN", location=(0, 0, CAM_H))
    sun = bpy.context.active_object
    # Tilt sun 15-25° for directional shadows and specular highlights on sticks
    tilt_angle = math.radians(random.uniform(15.0, 25.0))
    tilt_dir = random.uniform(0, 2 * math.pi)
    sun.rotation_euler = (tilt_angle, 0.0, tilt_dir)
    sun.data.energy = random.uniform(2.5, 4.0)
    sun.data.angle = math.radians(10)

    warm = random.uniform(0.95, 1.0)
    cool = random.uniform(0.95, 1.0)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (warm, 1.0, cool, 1.0)
        bg_node.inputs["Strength"].default_value = random.uniform(0.2, 0.35)

    return sun


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = ENGINE
    scene.render.resolution_x = W_OUT
    scene.render.resolution_y = H_OUT
    scene.render.resolution_percentage = 100

    if ENGINE == "BLENDER_WORKBENCH":
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "MATERIAL"
        shading.show_specular_highlight = False
        print("Workbench: solid renderer, material colours enabled")

    elif ENGINE == "CYCLES":
        samples = 64 if args.stick_style == "realistic" else SAMPLES
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        try:
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = 'GPU'
            print('Cycles: GPU rendering enabled')
        except Exception as e:
            print(f'Cycles: GPU not available, using CPU ({e})')
            scene.cycles.device = 'CPU'

    scene.frame_start = 1
    scene.frame_end = SIM_STEPS


def run_physics(stick_objects):
    """Advance simulation frame-by-frame, then bake final poses as keyframes.

    Keyframing the final pose at SIM_STEPS+1 (one frame past the sim end)
    overrides the physics cache completely for rendering — no cache drift,
    no ghost poses. This is the BlenderProc-validated approach.
    """
    scene = bpy.context.scene
    scene.frame_set(1)
    for frame in range(1, SIM_STEPS + 1):
        scene.frame_set(frame)

    # Read settled poses from matrix_world (the only property physics updates)
    scene.frame_set(SIM_STEPS)
    bpy.context.view_layer.update()

    final_poses = {}
    for obj in stick_objects:
        final_poses[obj.name] = obj.matrix_world.copy()

    # Bake: disable physics and keyframe the final pose at render frame
    render_frame = SIM_STEPS + 1
    scene.frame_set(render_frame)
    for obj in stick_objects:
        mat = final_poses[obj.name]
        obj.rigid_body.type = "PASSIVE"
        obj.location = mat.translation
        obj.rotation_euler = mat.to_euler()
        obj.keyframe_insert(data_path="location", frame=render_frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=render_frame)

    scene.frame_end = render_frame


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _clip_line_to_rect(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Clip a line segment to a rectangle using Cohen-Sutherland algorithm.

    Returns clipped (x1, y1, x2, y2) or None if entirely outside.
    """
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def _code(x, y):
        c = INSIDE
        if x < xmin:   c |= LEFT
        elif x > xmax: c |= RIGHT
        if y < ymin:   c |= BOTTOM
        elif y > ymax: c |= TOP
        return c

    c1, c2 = _code(x1, y1), _code(x2, y2)
    for _ in range(20):
        if not (c1 | c2):
            return x1, y1, x2, y2
        if c1 & c2:
            return None
        c = c1 or c2
        dx, dy = x2 - x1, y2 - y1
        if c & TOP:
            x = x1 + dx * (ymax - y1) / dy if dy else x1
            y = ymax
        elif c & BOTTOM:
            x = x1 + dx * (ymin - y1) / dy if dy else x1
            y = ymin
        elif c & RIGHT:
            y = y1 + dy * (xmax - x1) / dx if dx else y1
            x = xmax
        elif c & LEFT:
            y = y1 + dy * (xmin - x1) / dx if dx else y1
            x = xmin
        if c == c1:
            x1, y1, c1 = x, y, _code(x, y)
        else:
            x2, y2, c2 = x, y, _code(x, y)
    return None


def get_stick_obb_in_image(stick_obj, cam_obj, scene):
    """Return the 4 corners of the stick's OBB projected into image space (0-1).

    Projects stick endpoints to pixel space, clips the centerline to the
    image bounds, then expands by THICKNESS_PX/2 perpendicular pixels —
    matching the mikado-judge line_to_obb_corners convention.

    Returns None if the stick is completely off-screen.
    """
    from bpy_extras.object_utils import world_to_camera_view

    mat = stick_obj.matrix_world
    half_vec = mat.to_3x3() @ Vector((L / 2, 0, 0))
    center = mat.translation
    p1_world = center + half_vec
    p2_world = center - half_vec

    # Project endpoints to pixel coordinates
    v1 = world_to_camera_view(scene, cam_obj, p1_world)
    v2 = world_to_camera_view(scene, cam_obj, p2_world)
    x1, y1 = v1.x * W_OUT, (1.0 - v1.y) * H_OUT
    x2, y2 = v2.x * W_OUT, (1.0 - v2.y) * H_OUT

    # Clip the centerline to the image rectangle (with a small margin for
    # the perpendicular expansion so OBB corners don't poke out too far)
    margin = THICKNESS_PX / 2.0 + 1
    clipped = _clip_line_to_rect(
        x1, y1, x2, y2,
        -margin, -margin, W_OUT + margin, H_OUT + margin,
    )
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped

    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None

    # Perpendicular offset in pixels (same as mikado-judge line_to_obb_corners)
    theta = math.atan2(dy, dx)
    half_t = THICKNESS_PX / 2.0
    px = half_t * math.sin(theta)
    py = half_t * math.cos(theta)

    # 4 corners in pixel space (same winding as mikado-judge)
    corners_px = [
        (x1 - px, y1 + py),
        (x1 + px, y1 - py),
        (x2 + px, y2 - py),
        (x2 - px, y2 + py),
    ]

    # Normalise to 0-1, clamp any residual overshoot from perpendicular expansion
    corners = [
        (max(0.0, min(1.0, cx / W_OUT)), max(0.0, min(1.0, cy / H_OUT)))
        for cx, cy in corners_px
    ]

    return corners


def write_label(label_path, stick_objects, cam_obj, scene):
    lines = []
    for obj in stick_objects:
        class_name = obj.get("class_name")
        if class_name is None:
            continue
        class_id = NAME_TO_ID[class_name]
        corners = get_stick_obb_in_image(obj, cam_obj, scene)
        if corners is None:
            continue
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
        lines.append(f"{class_id} {coords}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_scene(index):
    scene = bpy.context.scene

    # Rigid body world — created fresh each scene (world_remove in clear_scene)
    bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    rbw.enabled = True
    rbw.substeps_per_frame = 20   # more substeps = less tunneling for thin sticks
    rbw.solver_iterations = 40

    # Table
    table_color = random.choice(RENDER["table"]["color_options"])
    add_table(table_color)

    # Camera and lighting
    cam = setup_camera()
    setup_lighting()

    # Build stick list — shuffle so colours are distributed randomly in the pile
    all_sticks = []
    for class_name, max_count in COUNTS.items():
        n = random.randint(max(1, max_count // 2), max_count)
        for _ in range(n):
            all_sticks.append(class_name)
    random.shuffle(all_sticks)

    stick_objects = []
    for i, class_name in enumerate(all_sticks):
        x = random.uniform(-DROP_W / 2, DROP_W / 2)
        y = random.uniform(-DROP_H / 2, DROP_H / 2)
        # Stagger heights: first stick at DROP_HEIGHT, last at 1.5×DROP_HEIGHT.
        # Smaller stagger keeps sticks from bouncing too far apart.
        z = DROP_HEIGHT + (i / max(len(all_sticks) - 1, 1)) * DROP_HEIGHT * 0.5
        # Near-horizontal spawn (±17°) matching a real mikado throw; rz is
        # the in-plane compass direction — fully random.
        rx = random.uniform(-0.3, 0.3)
        ry = random.uniform(-0.3, 0.3)
        rz = random.uniform(0, math.pi)
        if args.stick_style == "realistic":
            obj = add_stick_realistic(class_name, (x, y, z), (rx, ry, rz))
        else:
            obj = add_stick(class_name, (x, y, z), (rx, ry, rz))
        stick_objects.append(obj)

    # Run physics and freeze final poses
    run_physics(stick_objects)

    # Render
    img_name = f"synthetic_{index:05d}.png"
    img_path = os.path.join(images_dir, img_name)
    scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    # Write label
    lbl_path = os.path.join(labels_dir, f"synthetic_{index:05d}.txt")
    write_label(lbl_path, stick_objects, cam, scene)

    print(f"[{index+1}/{args.count}] Rendered {img_name} — {len(stick_objects)} sticks")


setup_render()

for i in range(args.count):
    clear_scene()
    generate_scene(i)

print(f"\nDone. Generated {args.count} images in {args.output}/")
