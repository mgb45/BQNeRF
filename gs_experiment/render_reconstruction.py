"""Render ground-truth vs. gsplat-reconstruction comparisons for a
trained checkpoint (gs_experiment.train_minimal_gsplat's output), plus
error maps -- a visual complement to differentiation_experiment.py's
uncertainty maps, so a reader can actually see what the reconstructed
scene looks like next to where BQ/visibility flag it as uncertain.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/render_reconstruction.py <scene_dir> [--view-indices 0 20 40] [--out <path>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def render_views(scene_dir: str, view_indices, background_color=(0.05, 0.05, 0.05), device="cuda"):
    import gsplat

    checkpoint = read_3dgs_ply(os.path.join(scene_dir, "splats.ply"))
    camera_angle_x, frames = load_transforms(os.path.join(scene_dir, "transforms.json"))

    with Image.open(os.path.join(scene_dir, frames[0][0] + ".png")) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(camera_angle_x, width, height)
    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    positions = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    rotations = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    # (N, 3, K) on disk (this project's convention, see ply_io); gsplat wants (N, K, 3)
    sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = checkpoint["sh_degree"]

    results = []
    with torch.no_grad():
        for i in view_indices:
            file_path, c2w = frames[i]
            viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device=device)[None]
            Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]

            rendered, _, _ = gsplat.rasterization(
                positions, rotations, scales, opacities, sh, viewmat, Ks,
                width=width, height=height, sh_degree=sh_degree, backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()

            gt_path = os.path.join(scene_dir, file_path + ".png")
            gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0

            results.append((i, gt, recon))
    return results, checkpoint


def plot_comparisons(results, out_path, title):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, (i, gt, recon) in enumerate(results):
        err = np.abs(gt - recon).mean(axis=-1)
        axes[row, 0].imshow(gt)
        axes[row, 0].set_title(f"view {i}: ground truth" if row == 0 else "")
        axes[row, 1].imshow(recon)
        axes[row, 1].set_title("gsplat reconstruction" if row == 0 else "")
        im = axes[row, 2].imshow(err, cmap="inferno", vmin=0, vmax=0.3)
        axes[row, 2].set_title("|error| (mean over RGB)" if row == 0 else "")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"view {i}", fontsize=9)
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir")
    parser.add_argument("--view-indices", type=int, nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--background-color", type=float, nargs=3, default=[0.05, 0.05, 0.05],
        help="must match the background the checkpoint was trained against (e.g. 1 1 1 for NeRF-Synthetic's white)",
    )
    args = parser.parse_args()

    _, frames = load_transforms(os.path.join(args.scene_dir, "transforms.json"))
    n_views = len(frames)
    view_indices = args.view_indices or sorted(set([0, n_views // 4, n_views // 2, n_views - 1]))
    view_indices = [i for i in view_indices if 0 <= i < n_views]

    results, checkpoint = render_views(args.scene_dir, view_indices, background_color=tuple(args.background_color))

    psnrs = []
    for i, gt, recon in results:
        mse = float(np.mean((gt - recon) ** 2))
        psnr = -10.0 * np.log10(max(mse, 1e-10))
        psnrs.append(psnr)
        print(f"view {i}: PSNR {psnr:.2f}dB")
    print(f"mean PSNR over shown views: {np.mean(psnrs):.2f}dB  ({checkpoint['positions'].shape[0]} splats)")

    out_path = args.out or (RESULTS_DIR / f"reconstruction_{Path(args.scene_dir).name}.png")
    plot_comparisons(results, out_path, title=f"gsplat reconstruction vs. ground truth ({args.scene_dir})")


if __name__ == "__main__":
    main()
