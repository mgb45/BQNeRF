"""Read/write NeRF-synthetic-style `transforms.json` (camera_angle_x +
per-frame camera-to-world matrices in OpenGL convention: x right, y up, z
back, since the camera looks down its own local -Z), and convert those
poses to the two conventions the rest of this project needs:
gs_experiment.camera.CameraPose (world-space center/forward/up, used by
the BQ/visibility pipeline) and gsplat's OpenCV-convention world-to-camera
matrix (x right, y down, z forward -- used at the rasterization boundary).

Pure numpy + json, no torch/bpy dependency, so the coordinate-conversion
logic is testable without Blender or a GPU.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

from gs_experiment.camera import CameraPose


def write_transforms_json(path: str, camera_angle_x: float, frames: List[dict]) -> None:
    """frames: list of {"file_path": str, "transform_matrix": (4,4) array-like}."""
    payload = {
        "camera_angle_x": float(camera_angle_x),
        "frames": [
            {"file_path": f["file_path"], "transform_matrix": np.asarray(f["transform_matrix"]).tolist()}
            for f in frames
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_transforms(path: str) -> Tuple[float, List[Tuple[str, np.ndarray]]]:
    """Returns (camera_angle_x_radians, [(file_path, c2w_4x4), ...])."""
    with open(path) as fh:
        payload = json.load(fh)
    camera_angle_x = float(payload["camera_angle_x"])
    frames = [(frame["file_path"], np.array(frame["transform_matrix"], dtype=float)) for frame in payload["frames"]]
    return camera_angle_x, frames


def camera_pose_from_c2w(c2w: np.ndarray) -> CameraPose:
    """c2w columns are (right, up, back, translation) in world space --
    'back' because the camera looks down its local -Z (OpenGL/NeRF-
    synthetic convention)."""
    center = c2w[:3, 3]
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    back = c2w[:3, 2]
    forward = -back
    return CameraPose(
        center=center,
        forward=forward / np.linalg.norm(forward),
        up=up / np.linalg.norm(up),
    )


def opencv_viewmat_from_c2w(c2w: np.ndarray) -> np.ndarray:
    """gsplat/OpenCV convention is x right, y down, z forward (into the
    scene) -- world-to-camera. NeRF-synthetic's c2w is OpenGL convention
    (x right, y up, z back); negating the y/z basis columns converts
    between the two before inverting to world-to-camera. The translation
    column (index 3) is untouched by this, so the camera center is
    unaffected -- only the axis directions flip."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_cv = c2w @ flip
    return np.linalg.inv(c2w_cv)


def fov_x_to_intrinsics(camera_angle_x: float, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics matrix from a horizontal FOV (radians) and image
    size, assuming a centered principal point and square pixels."""
    fx = (width / 2.0) / np.tan(camera_angle_x / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
