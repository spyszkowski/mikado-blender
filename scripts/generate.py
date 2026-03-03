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
    # Rigid body — passive (static floor)
    bpy.ops.rigidbody.object_add()
    table.rigid_body.type = "PASSIVE"
    table.rigid_body.collision_shape = "MESH"
    return table


def add_stick(class_name, location, rotation_euler):
    """Create a stick as a cylinder with coloured tip caps."""
    # Main body
    bpy.ops.mesh.primitive_cylinder_add(
        radius=D / 2,
        depth=L,
        location=location,
        rotation=rotation_euler,
    )
    stick = bpy.context.active_object
    stick.name = f"stick_{class_name}_{random.randint(0, 99999)}"

    body_mat = make_material(f"body_{stick.name}", BODY_COLOR, roughness=0.4)
    stick.data.materials.append(body_mat)

    tip_color = TIP_COLORS[class_name]

    # Tip caps — small cylinders at each end, same axis
    tip_len = L * TIP_FRAC
    for sign in (+1, -1):
        offset = Vector((0, 0, sign * (L / 2 - tip_len / 2)))
        # Transform offset into world space using stick's rotation
        rot_mat = stick.rotation_euler.to_matrix()
        world_offset = rot_mat @ offset
        tip_loc = Vector(location) + world_offset

        bpy.ops.mesh.primitive_cylinder_add(
            radius=D / 2 + 0.0001,   # tiny overlap to avoid z-fighting
            depth=tip_len,
            location=tip_loc,
            rotation=rotation_euler,
        )
        tip = bpy.context.active_object
        tip.name = f"tip_{stick.name}_{sign}"
        tip_mat = make_material(f"tipmat_{tip.name}", tip_color, roughness=0.3)
        tip.data.materials.append(tip_mat)

        # Join tip to stick
        tip.select_set(True)
        stick.select_set(True)
        bpy.context.view_layer.objects.active = stick
        bpy.ops.object.join()

    # Rigid body — active (falls under gravity)
    bpy.ops.rigidbody.object_add()
    stick.rigid_body.type = "ACTIVE"
    stick.rigid_body.collision_shape = "CAPSULE"
    stick.rigid_body.mass = 0.005   # 5g per stick
    stick.rigid_body.restitution = 0.1
    stick.rigid_body.friction = 0.8

    stick["class_name"] = class_name
    return stick


def setup_camera():
    bpy.ops.object.camera_add(location=(0, 0, CAM_H))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (0, 0, 0)   # looking straight down
    bpy.context.scene.camera = cam

    fov = math.radians(RENDER["camera"]["fov_deg"])
    cam.data.lens_unit = "FOV"
    cam.data.angle = fov
    return cam


def setup_lighting():
    cfg = RENDER["lighting"]
    energy = random.uniform(cfg["min_energy"], cfg["max_energy"])
    jitter = cfg["jitter_mm"] / 1000.0
    loc = (
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
        CAM_H * 0.8,
    )
    bpy.ops.object.light_add(type="POINT", location=loc)
    light = bpy.context.active_object
    light.data.energy = energy
    return light


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = ENGINE
    scene.render.resolution_x = W_OUT
    scene.render.resolution_y = H_OUT
    scene.render.resolution_percentage = 100
    if ENGINE == "CYCLES":
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True
    scene.frame_start = 1
    scene.frame_end = SIM_STEPS


def run_physics():
    """Advance simulation to let sticks settle."""
    scene = bpy.context.scene
    scene.frame_set(1)
    for frame in range(1, SIM_STEPS + 1):
        scene.frame_set(frame)


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def get_stick_obb_in_image(stick_obj, cam_obj, scene):
    """Return the 4 corners of the stick's OBB projected into image space (0-1).

    Returns None if the stick is not visible (off-screen).
    """
    from bpy_extras.object_utils import world_to_camera_view

    # Get the world-space endpoints of the stick's long axis
    mat = stick_obj.matrix_world
    half = Vector((0, 0, L / 2))
    p1_world = mat @ half
    p2_world = mat @ (-half)

    # Project to camera/image space (0-1, y-up in Blender → flip y for image)
    def proj(p):
        v = world_to_camera_view(scene, cam_obj, p)
        return (v.x, 1.0 - v.y)   # flip y: Blender y=0 is bottom, image y=0 is top

    p1 = proj(p1_world)
    p2 = proj(p2_world)

    # Build OBB corners from the line + thickness
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None

    # Half-thickness in normalised image coordinates
    # D (metres) projected: approximate using stick length ratio
    stick_len_img = length
    stick_len_real = L
    half_t = (D / stick_len_real) * stick_len_img / 2.0

    # Perpendicular unit vector
    px = -dy / length * half_t
    py = dx / length * half_t

    corners = [
        (p1[0] - px, p1[1] - py),
        (p1[0] + px, p1[1] + py),
        (p2[0] + px, p2[1] + py),
        (p2[0] - px, p2[1] - py),
    ]

    # Check at least partially in frame
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    if max(xs) < 0 or min(xs) > 1 or max(ys) < 0 or min(ys) > 1:
        return None

    # Clamp to [0, 1]
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

    # Enable rigid body world
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    # Blender 3.x uses substeps_per_frame instead of steps_per_second
    if hasattr(rbw, 'steps_per_second'):
        rbw.steps_per_second = 120
    if hasattr(rbw, 'substeps_per_frame'):
        rbw.substeps_per_frame = 10
    if hasattr(rbw, 'solver_iterations'):
        rbw.solver_iterations = 20

    # Table
    table_color = random.choice(RENDER["table"]["color_options"])
    add_table(table_color)

    # Camera and lighting
    cam = setup_camera()
    setup_lighting()

    # Determine how many sticks of each type to place
    stick_objects = []
    for class_name, max_count in COUNTS.items():
        n = random.randint(max(1, max_count // 2), max_count)
        for _ in range(n):
            x = random.uniform(-DROP_W / 2, DROP_W / 2)
            y = random.uniform(-DROP_H / 2, DROP_H / 2)
            z = DROP_HEIGHT + random.uniform(0, DROP_HEIGHT * 0.5)
            # Random rotation — mostly horizontal with slight tilt
            rx = random.uniform(-0.2, 0.2)
            ry = random.uniform(-0.2, 0.2)
            rz = random.uniform(0, math.pi)
            obj = add_stick(class_name, (x, y, z), (rx, ry, rz))
            stick_objects.append(obj)

    # Run physics
    run_physics()
    scene.frame_set(SIM_STEPS)

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
