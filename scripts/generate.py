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

W_OUT = RENDER["output"]["width"]
H_OUT = RENDER["output"]["height"]
SAMPLES = RENDER["output"]["samples"]
ENGINE = RENDER["output"]["engine"]

CAM_H = RENDER["camera"]["height_mm"] / 1000.0
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


def make_wood_material(name, base_color, roughness=0.5):
    """Procedural bamboo/wood material with longitudinal grain and specular highlights."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    for n in nodes:
        nodes.remove(n)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value = roughness
    # "Specular IOR Level" (Blender 4.0+) was previously called "Specular"
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.6
            break
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)

    # Wave texture — longitudinal grain along X (stick long axis)
    wave = nodes.new("ShaderNodeTexWave")
    wave.location = (-500, 200)
    wave.wave_type = "BANDS"
    if hasattr(wave, "bands_direction"):  # Blender 3.1+
        wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = random.uniform(40.0, 80.0)
    wave.inputs["Distortion"].default_value = random.uniform(2.0, 4.0)
    wave.inputs["Detail"].default_value = 4.0
    if "Detail Scale" in wave.inputs:  # Blender 3.0+
        wave.inputs["Detail Scale"].default_value = 2.0
    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    # Fine noise for grain micro-variation
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-500, -50)
    noise.inputs["Scale"].default_value = random.uniform(150.0, 300.0)
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.7
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    # Mix wave + noise
    mix_grain = nodes.new("ShaderNodeMixRGB")
    mix_grain.location = (-250, 100)
    mix_grain.inputs["Fac"].default_value = 0.3
    links.new(wave.outputs["Fac"], mix_grain.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix_grain.inputs["Color2"])

    # Color ramp: base wood color → slightly darker grain lines
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-50, 100)
    r, g, b = base_color
    darken = 0.82
    lighten = 1.08
    color_ramp.color_ramp.elements[0].position = 0.3
    color_ramp.color_ramp.elements[0].color = (
        min(1, r * lighten), min(1, g * lighten), min(1, b * lighten), 1.0)
    color_ramp.color_ramp.elements[1].position = 0.7
    color_ramp.color_ramp.elements[1].color = (r * darken, g * darken, b * darken, 1.0)
    links.new(mix_grain.outputs["Color"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Bump from grain
    bump = nodes.new("ShaderNodeBump")
    bump.location = (100, -150)
    bump.inputs["Strength"].default_value = random.uniform(0.01, 0.03)
    bump.inputs["Distance"].default_value = 0.05
    links.new(mix_grain.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def make_tip_material_realistic(name, tip_color, roughness=0.3):
    """Painted tip material with ring-stripe bands and slight imperfections."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    for n in nodes:
        nodes.remove(n)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)

    # Wave texture — ring stripes around the stick (bands along X)
    wave = nodes.new("ShaderNodeTexWave")
    wave.location = (-500, 200)
    wave.wave_type = "BANDS"
    if hasattr(wave, "bands_direction"):  # Blender 3.1+
        wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = random.uniform(300.0, 500.0)
    wave.inputs["Distortion"].default_value = random.uniform(0.5, 1.5)
    wave.inputs["Detail"].default_value = 2.0
    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    # Noise for paint imperfections
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-500, -50)
    noise.inputs["Scale"].default_value = random.uniform(150.0, 250.0)
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.5
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    # Color ramp: tip color → slightly shifted variant for ring lines
    r, g, b = tip_color
    darken = 0.75
    lighten = 1.15

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-50, 200)
    color_ramp.color_ramp.elements[0].position = 0.35
    color_ramp.color_ramp.elements[0].color = (
        min(1, r * lighten), min(1, g * lighten), min(1, b * lighten), 1.0)
    color_ramp.color_ramp.elements[1].position = 0.65
    color_ramp.color_ramp.elements[1].color = (r * darken, g * darken, b * darken, 1.0)

    # Mix wave + noise to drive color ramp
    mix_paint = nodes.new("ShaderNodeMixRGB")
    mix_paint.location = (-250, 100)
    mix_paint.inputs["Fac"].default_value = 0.15  # mostly wave stripes, slight noise
    links.new(wave.outputs["Fac"], mix_paint.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix_paint.inputs["Color2"])
    links.new(mix_paint.outputs["Color"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Subtle bump from wave for paint texture
    bump = nodes.new("ShaderNodeBump")
    bump.location = (100, -150)
    bump.inputs["Strength"].default_value = 0.005
    bump.inputs["Distance"].default_value = 0.02
    links.new(wave.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def add_stick_realistic(class_name, location, rotation_euler):
    """Create a stick with wood grain body and ring-stripe tips.

    Same geometry and physics as add_stick(), but with procedural materials
    that simulate real bamboo/wood texture and painted tip bands.
    """
    tip_len = L * TIP_FRAC
    segs = 12  # higher segment count for smoother specular highlights

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

    # Wood grain body material with per-stick color jitter
    r, g, b = BODY_COLOR
    jitter = 0.06
    wood_color = (
        max(0, min(1, r + random.uniform(-jitter, jitter))),
        max(0, min(1, g + random.uniform(-jitter, jitter))),
        max(0, min(1, b + random.uniform(-jitter, jitter))),
    )
    obj.data.materials.append(make_wood_material(
        f"body_{obj.name}", wood_color, roughness=random.uniform(0.4, 0.6)))
    obj.data.materials.append(make_tip_material_realistic(
        f"tip_{obj.name}", TIP_COLORS[class_name], roughness=random.uniform(0.25, 0.4)))

    obj.location = location
    obj.rotation_euler = rotation_euler

    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "ACTIVE"
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.001
    obj.rigid_body.mass = 0.005
    obj.rigid_body.restitution = 0.2
    obj.rigid_body.friction = 0.6
    obj.rigid_body.angular_damping = 0.6
    obj.rigid_body.linear_damping = 0.6

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
    obj.rigid_body.restitution = 0.2
    obj.rigid_body.friction = 0.6
    obj.rigid_body.angular_damping = 0.6   # damp rotation so sticks settle quickly
    obj.rigid_body.linear_damping = 0.6    # damp translation so sticks stop sliding

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
    sun.rotation_euler = (0.0, 0.0, 0.0)
    sun.data.energy = random.uniform(1.0, 2.0)
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
        bg_node.inputs["Strength"].default_value = random.uniform(0.4, 0.6)

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

def get_stick_obb_in_image(stick_obj, cam_obj, scene):
    """Return the 4 corners of the stick's OBB projected into image space (0-1).

    Returns None if the stick is not visible (off-screen).
    """
    from bpy_extras.object_utils import world_to_camera_view

    # After transform_apply the mesh long axis is local Z.
    # matrix_world is kept current by physics (and frozen above).
    mat = stick_obj.matrix_world
    half_vec = mat.to_3x3() @ Vector((L / 2, 0, 0))  # long axis is local X
    center = mat.translation
    p1_world = center + half_vec
    p2_world = center - half_vec

    def proj(p):
        v = world_to_camera_view(scene, cam_obj, p)
        return (v.x, 1.0 - v.y)

    p1 = proj(p1_world)
    p2 = proj(p2_world)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None

    stick_len_img = length
    half_t = (D / L) * stick_len_img / 2.0

    px = -dy / length * half_t
    py = dx / length * half_t

    corners = [
        (p1[0] - px, p1[1] - py),
        (p1[0] + px, p1[1] + py),
        (p2[0] + px, p2[1] + py),
        (p2[0] - px, p2[1] - py),
    ]

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    if max(xs) < 0 or min(xs) > 1 or max(ys) < 0 or min(ys) > 1:
        return None

    corners = [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in corners]
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
        # Stagger heights: first stick at DROP_HEIGHT, last at 2×DROP_HEIGHT.
        # Sequential landing forces sticks to rest on the growing pile.
        z = DROP_HEIGHT + (i / max(len(all_sticks) - 1, 1)) * DROP_HEIGHT
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
