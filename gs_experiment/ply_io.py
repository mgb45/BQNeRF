"""Read/write the standard 3D Gaussian Splatting .ply checkpoint schema
(the point-cloud format Inria's original 3DGS implementation and
gsplat-based trainers both use for saved splats), via plyfile. Kept as
its own module (numpy + plyfile only, no torch) so
gs_experiment.splat_scene's loader can use it without depending on the
trainer, and gs_experiment.train_minimal_gsplat's writer can use it
without the loader depending on torch.

Property order and semantics match the reference implementation:
- position: x, y, z (raw, not offsets)
- normal: nx, ny, nz (written as zeros -- 3DGS doesn't train normals, the
  field exists only for point-cloud-viewer compatibility)
- f_dc_0..2: degree-0 SH coefficient per channel (view-independent base
  color term, NOT the color itself -- see spherical_harmonics.eval_sh's
  "+0.5" offset convention)
- f_rest_*: higher-degree SH coefficients, channel-major flattening of a
  (3, n_coeffs_for_degree - 1) array: f_rest_{c * n_rest + k} is channel
  c's k-th higher-order coefficient. This channel-major order (rather
  than coefficient-major) is what the reference implementation's
  `features_rest.transpose(1, 2)` produces -- matched here for interop
  with real checkpoints, not just internal consistency.
- opacity: stored as the pre-sigmoid logit, not the [0, 1] opacity itself
  (matches 3DGS's inverse_sigmoid activation)
- scale_0..2: stored as log(scale), not the scale itself (matches 3DGS's
  exp activation)
- rot_0..3: quaternion in (w, x, y, z) order, as trained (not necessarily
  unit-norm in the file -- normalized on read)
"""

from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE


def _inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


def write_3dgs_ply(
    path: str,
    positions: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    sh_coeffs: np.ndarray,
    sh_degree: int,
) -> None:
    """positions (N,3); scales (N,3) as real (positive) scales; rotations
    (N,4) quaternions (need not be unit-norm); opacities (N,) in (0, 1);
    sh_coeffs (N, 3, n_coeffs_for(sh_degree))."""
    n = positions.shape[0]
    n_coeffs = N_COEFFS_FOR_DEGREE[sh_degree]
    n_rest = n_coeffs - 1

    f_dc = sh_coeffs[:, :, 0]
    f_rest = sh_coeffs[:, :, 1:].reshape(n, 3 * n_rest) if n_rest > 0 else np.zeros((n, 0))

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(3 * n_rest)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]

    data = np.concatenate(
        [
            positions,
            np.zeros((n, 3)),
            f_dc,
            f_rest,
            _inverse_sigmoid(np.clip(opacities, 1e-6, 1 - 1e-6))[:, None],
            np.log(scales),
            rotations,
        ],
        axis=1,
    ).astype(np.float32)

    dtype = [(name, "f4") for name in names]
    structured = np.empty(n, dtype=dtype)
    for i, name in enumerate(names):
        structured[name] = data[:, i]

    element = PlyElement.describe(structured, "vertex")
    PlyData([element]).write(path)


def read_3dgs_ply(path: str) -> dict:
    """Returns dict with positions (N,3), scales (N,3) [real, exp'd],
    rotations (N,4) [normalized], opacities (N,) [sigmoid'd, in (0, 1)],
    sh_coeffs (N, 3, n_coeffs), sh_degree (int)."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    n = v.count

    positions = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(float)

    f_dc = np.stack([np.asarray(v[f"f_dc_{i}"]) for i in range(3)], axis=1).astype(float)

    rest_names = sorted(
        (name for name in v.data.dtype.names if name.startswith("f_rest_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    n_rest = len(rest_names) // 3
    n_coeffs = n_rest + 1
    degree_for_n_coeffs = {n_c: deg for deg, n_c in N_COEFFS_FOR_DEGREE.items()}
    if n_coeffs not in degree_for_n_coeffs:
        raise ValueError(f"ply has {n_coeffs} SH coefficients/channel, not a supported degree (0-3)")
    sh_degree = degree_for_n_coeffs[n_coeffs]

    if n_rest > 0:
        f_rest = np.stack([np.asarray(v[name]) for name in rest_names], axis=1).astype(float)
        f_rest = f_rest.reshape(n, 3, n_rest)
    else:
        f_rest = np.zeros((n, 3, 0))

    sh_coeffs = np.concatenate([f_dc[:, :, None], f_rest], axis=2)

    opacities = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"]).astype(float)))

    scales = np.exp(np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)], axis=1).astype(float))

    rotations = np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)], axis=1).astype(float)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    return dict(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
        sh_degree=sh_degree,
    )
