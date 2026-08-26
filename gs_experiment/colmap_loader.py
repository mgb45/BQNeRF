"""Reads COLMAP's binary reconstruction format (`cameras.bin`,
`images.bin` -- the standard output of COLMAP's sparse reconstruction,
and how Mip-NeRF360/Tanks & Temples-style real-captured datasets ship
their camera poses) and converts to this project's `transforms.json`
convention (`nerf_transforms.load_transforms`'s OpenGL c2w format).

Every prior scene in this project either had hand-authored ground-truth
poses (Blender scenes via `scene_spec`/`blender_render`) or NeRF-
Synthetic's own pre-baked `transforms.json`. COLMAP is the first real,
estimated-from-photographs pose source this project reads -- poses here
come from structure-from-motion on real images, not a known ground-truth
rig.

Format reference: COLMAP's own `scripts/python/read_write_model.py`
(the de facto spec for this binary layout; not vendored here, re-
implemented directly from the documented struct layout since only
`cameras.bin`/`images.bin` reading is needed, not COLMAP's full read/write
API surface).

Known, stated approximation: COLMAP camera models can include lens
distortion parameters (`SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, ...); this
project's rendering pipeline (`nerf_transforms.fov_x_to_intrinsics`,
gsplat's pinhole rasterizer) has no distortion model, so distortion
coefficients are read (for the caller to inspect/warn on) but not applied
-- real photos are assumed close enough to an ideal pinhole for this
project's purposes, not undistorted. A real, acknowledged source of
pose/geometry error for a real capture, not present in any prior scene.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

CAMERA_MODEL_NUM_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-specific; params[:2] is (fx, fy) or (f, f) depending on model


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray  # (4,) COLMAP world-to-camera rotation, (qw, qx, qy, qz)
    tvec: np.ndarray  # (3,) COLMAP world-to-camera translation
    camera_id: int
    name: str


def _read_next_bytes(fh, num_bytes, fmt):
    data = fh.read(num_bytes)
    return struct.unpack(fmt, data)


def read_cameras_binary(path: str) -> Dict[int, ColmapCamera]:
    cameras = {}
    with open(path, "rb") as fh:
        (num_cameras,) = _read_next_bytes(fh, 8, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fh, 24, "<iiQQ")
            model_name, num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(_read_next_bytes(fh, 8 * num_params, "<" + "d" * num_params))
            cameras[camera_id] = ColmapCamera(camera_id=camera_id, model=model_name, width=width, height=height, params=params)
    return cameras


def read_images_binary(path: str) -> Dict[int, ColmapImage]:
    images = {}
    with open(path, "rb") as fh:
        (num_reg_images,) = _read_next_bytes(fh, 8, "<Q")
        for _ in range(num_reg_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read_next_bytes(fh, 64, "<idddddddi")
            name_chars = []
            while True:
                (c,) = _read_next_bytes(fh, 1, "<c")
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")
            (num_points2d,) = _read_next_bytes(fh, 8, "<Q")
            fh.read(24 * num_points2d)  # x, y (double), point3D_id (int64) per point -- not needed here
            images[image_id] = ColmapImage(
                image_id=image_id, qvec=np.array([qw, qx, qy, qz]), tvec=np.array([tx, ty, tz]), camera_id=camera_id, name=name,
            )
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def colmap_image_to_c2w_opengl(image: ColmapImage) -> np.ndarray:
    """COLMAP's (qvec, tvec) is a world-to-camera transform in an OpenCV-
    style convention (x right, y down, z forward) -- the same convention
    `nerf_transforms.opencv_viewmat_from_c2w` produces as its *output*, so
    this is that function's exact inverse operation: invert to get
    camera-to-world in OpenCV convention, then flip the y/z basis columns
    to match this project's OpenGL-convention transforms.json (the same
    `flip = diag([1,-1,-1,1])` both functions share).
    """
    R = qvec_to_rotmat(image.qvec)
    t = image.tvec
    w2c_cv = np.eye(4)
    w2c_cv[:3, :3] = R
    w2c_cv[:3, 3] = t
    c2w_cv = np.linalg.inv(w2c_cv)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_cv @ flip


def camera_angle_x_from_camera(camera: ColmapCamera) -> float:
    """Horizontal FOV in radians from a COLMAP camera's focal length --
    this project's transforms.json convention assumes one shared,
    centered-principal-point pinhole intrinsics per scene
    (`fov_x_to_intrinsics`), so this is only exact when every image in
    the scene shares one `camera_id` (checked by the caller) and that
    camera's principal point is close to centered (real calibrations
    usually are, not verified here -- a real, stated approximation)."""
    fx = camera.params[0]
    return 2.0 * np.arctan((camera.width / 2.0) / fx)


def load_colmap_scene(sparse_dir: str) -> Tuple[float, List[Tuple[str, np.ndarray]]]:
    """Reads `<sparse_dir>/cameras.bin` + `images.bin`, returns
    (camera_angle_x_radians, [(file_path_without_extension, c2w_opengl), ...])
    -- the exact shape `nerf_transforms.load_transforms` returns, so every
    downstream consumer of that function works unchanged on a real COLMAP
    scene.

    Requires every image to share the same `camera_id` (checked, not
    assumed) -- true for every Mip-NeRF360/Tanks-&-Temples-style scene
    shot with one physical camera, which is what this project's single
    shared `camera_angle_x` convention needs.
    """
    import os

    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))

    camera_ids = {img.camera_id for img in images.values()}
    if len(camera_ids) != 1:
        raise ValueError(f"expected exactly one shared camera_id across all images, found {camera_ids}")
    camera = cameras[next(iter(camera_ids))]
    camera_angle_x = camera_angle_x_from_camera(camera)

    frames = []
    for image in sorted(images.values(), key=lambda im: im.name):
        c2w = colmap_image_to_c2w_opengl(image)
        file_stem = os.path.splitext(image.name)[0]
        frames.append((file_stem, c2w))

    return camera_angle_x, frames
