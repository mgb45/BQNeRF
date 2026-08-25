"""Blender headless renderer consuming a gs_experiment.scene_spec
RenderSceneSpec (as JSON) and producing a NeRF-synthetic-style dataset:
PNG images plus a transforms.json in gs_experiment.nerf_transforms'
schema (camera_angle_x + per-frame camera-to-world matrices, OpenGL
convention).

Needs `bpy` -- only importable/runnable inside a Blender process, so this
is a script, not a library module the rest of this package imports:

    blender --background --python gs_experiment/blender_render.py -- \
        <scene_spec.json> <output_dir>

Objects are unlit (Emission shader, not Principled BSDF) so color is
exactly the spec's `color`, independent of a light rig -- there is no
light rig, deliberately, to keep the resulting images easy for the
minimal gsplat trainer (gs_experiment/train_minimal_gsplat.py) to fit
without having to also learn plausible-looking specular/shading
variation that a from-scratch trainer isn't equipped to reconstruct
faithfully.
"""

import json
import math
import os
import sys

import bpy
import mathutils


def _load_spec(path):
    with open(path) as fh:
        return json.load(fh)


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _add_object(spec):
    obj_type = spec["type"]
    location = spec["location"]

    if obj_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(location=location, segments=32, ring_count=16)
    elif obj_type == "cube":
        bpy.ops.mesh.primitive_cube_add(location=location)
    elif obj_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(location=location, vertices=32)
    elif obj_type == "plane":
        bpy.ops.mesh.primitive_plane_add(location=location)
    else:
        raise ValueError(f"unknown object type: {obj_type}")

    obj = bpy.context.active_object
    obj.scale = spec["scale"]
    obj.rotation_euler = spec.get("rotation_euler", (0.0, 0.0, 0.0))

    color = spec["color"]
    mat = bpy.data.materials.new(name=f"mat_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.append(mat)


def _setup_world_background(color):
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (*color, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def _setup_render(resolution, fov_deg):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(fov_deg)
    cam_obj = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False  # this Blender build has no OpenImageDenoiser; unlit emission shading doesn't need it

    # CPU, deliberately: Cycles' CUDA kernel needs its own nvcc JIT compile
    # (separate from, and just as version-sensitive as, gsplat/torch's --
    # see requirements-gsplat.txt's build notes), and at this scene's
    # scale (dozens of small, low-sample-count images) CPU rendering is
    # fast enough that it isn't worth also solving that for Blender. The
    # GPU is spent on gsplat training instead, where it actually matters.
    scene.cycles.device = "CPU"

    return cam_obj


def _set_camera_pose(cam_obj, center, forward, up):
    """Same right/up derivation as gs_experiment.camera.camera_local_frame
    (right = forward x up, re-orthogonalized up = right x forward), then
    columns [right, up, -forward] for Blender/OpenGL's "camera looks down
    local -Z" convention."""
    forward_v = mathutils.Vector(forward).normalized()
    up_hint = mathutils.Vector(up).normalized()
    right = forward_v.cross(up_hint).normalized()
    true_up = right.cross(forward_v).normalized()
    back = -forward_v

    rot = mathutils.Matrix(
        (
            (right.x, true_up.x, back.x),
            (right.y, true_up.y, back.y),
            (right.z, true_up.z, back.z),
        )
    ).to_4x4()
    cam_obj.matrix_world = mathutils.Matrix.Translation(mathutils.Vector(center)) @ rot


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    spec_path, output_dir = argv[0], argv[1]
    spec = _load_spec(spec_path)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    _clear_scene()
    for obj_spec in spec["objects"]:
        _add_object(obj_spec)
    _setup_world_background(spec.get("background_color", [0.05, 0.05, 0.05]))
    cam_obj = _setup_render(spec["resolution"], spec["fov_deg"])

    frames = []
    for i, cam_spec in enumerate(spec["cameras"]):
        _set_camera_pose(cam_obj, cam_spec["center"], cam_spec["forward"], cam_spec["up"])
        file_stem = f"r_{i:03d}"
        bpy.context.scene.render.filepath = os.path.join(images_dir, file_stem + ".png")
        bpy.ops.render.render(write_still=True)

        c2w = [list(row) for row in cam_obj.matrix_world]
        frames.append({"file_path": f"images/{file_stem}", "transform_matrix": c2w})
        print(f"blender_render: rendered {i + 1}/{len(spec['cameras'])}")

    transforms = {"camera_angle_x": math.radians(spec["fov_deg"]), "frames": frames}
    with open(os.path.join(output_dir, "transforms.json"), "w") as fh:
        json.dump(transforms, fh, indent=2)
    print(f"blender_render: wrote {len(frames)} frames to {output_dir}")


if __name__ == "__main__":
    main()
