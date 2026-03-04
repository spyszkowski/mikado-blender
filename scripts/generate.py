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


def add_table(color):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    table = bpy.context.active_object
    table.name = "Table"
    mat = make_material("TableMat", color, roughness=0.8)
    table.data.materials.append(mat)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.rigidbody.object_add()
    table.rigid_body.type = "PASSIVE"
    table.rigid_body.collision_shape = "MESH"
    return table


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
    obj.rigid_body.mass = 0.005
    obj.rigid_body.restitution = 0.4
    obj.rigid_body.friction = 0.4
    obj.rigid_body.angular_damping = 0.1
    obj.rigid_body.linear_damping = 0.1

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
        scene.cycles.samples = SAMPLES
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
    """Advance simulation frame-by-frame to populate the physics cache,
    then freeze each stick's final pose so rendering reads the settled position."""
    scene = bpy.context.scene
    scene.frame_set(1)
    for frame in range(1, SIM_STEPS + 1):
        scene.frame_set(frame)

    # Freeze poses: read matrix_world (updated by physics) and apply as static
    # location/rotation so the render is not affected by any cache drift.
    # obj.location is NOT updated during simulation — only matrix_world is.
    scene.frame_set(SIM_STEPS)
    bpy.context.view_layer.update()
    for obj in stick_objects:
        mat = obj.matrix_world.copy()
        obj.rigid_body.type = "PASSIVE"   # stop physics overriding the pose
        obj.location = mat.translation
        obj.rotation_euler = mat.to_euler()
        obj.matrix_world = mat


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
