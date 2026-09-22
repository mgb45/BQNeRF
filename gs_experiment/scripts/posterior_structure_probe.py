"""How structured is the per-splat precision? The question that decides how
fast the uncertainty can be made.

Per-pixel variance has a closed form, because the render is linear in the SH
coefficients and the posterior is per-splat independent:

    Var[C(q)] = sum_i beta_{q,i}^2 * v_i,    v_i = phi(d)^T Sigma_i phi(d)

so a rasterizer that accumulates `v_i * vis * vis` alongside the colour gets
uncertainty for one extra fused multiply-add per (pixel, splat) -- no probes,
no draws, exact. The cost is then entirely in `v_i`, and `Sigma_i` is a
16x16 inverse per splat: 136 floats to store exactly, 550 MB per million
splats, which is not shippable.

Two approximations would be:

* **band-diagonal** -- `D_i ~ (w_i / 4pi) I`, one scalar per splat. By the
  spherical harmonic addition theorem `sum_m Y_lm(d)^2` is constant, so `v_i`
  becomes DIRECTION-INDEPENDENT: one float per splat, no per-frame work at
  all. It also throws away the view-dependence, which is the physically
  interesting part -- a splat seen from a narrow cone is determined there and
  nowhere else.
* **rank-1** -- `D_i ~ w_i phi(dbar_i) phi(dbar_i)^T` for the beta^2-weighted
  mean viewing direction `dbar_i`. Four floats per splat, keeps the
  view-dependence, and Sherman-Morrison gives `v_i(d)` from one dot product.

Which is honest is an empirical question about the spectrum of `D_i`, and
this measures it rather than assuming. Nothing is optimised until it answers.

Run: .venv-gsplat/bin/python gs_experiment/scripts/posterior_structure_probe.py -m MODEL -s SRC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "GS-U"))

import numpy as np
import torch
from PIL import Image

from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w
from gs_experiment.rasterized_sh_precision import (
    _sh_basis_torch, probe_squared_footprint_weights)
from gs_experiment.scripts.our_uncertainty_for_3dgs_model import (
    _FLIP, c2w_opencv, intrinsics, load_cameras)
from gs_experiment.scripts.render_posterior_ensemble import empirical_band_precision


def accumulate_with_moments(ck, frames, K_np, width, height, degree, n_probes, device):
    """The usual precision accumulation, plus the two moments a rank-1 model
    needs: the total beta^2 weight per splat and its weighted mean viewing
    direction. Both are free here -- they are sums over the same loop."""
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    means, quats = t(ck["positions"]), t(ck["rotations"])
    scales, opac = t(ck["scales"]), t(ck["opacities"])
    Ks = t(K_np)[None]
    n, kdim = means.shape[0], (degree + 1) ** 2

    D = torch.zeros((n, kdim, kdim), dtype=torch.float32, device=device)
    w_tot = torch.zeros(n, dtype=torch.float32, device=device)
    d_sum = torch.zeros((n, 3), dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    for p, (_, c2w) in enumerate(frames):
        gen.manual_seed(p)
        vm = t(opencv_viewmat_from_c2w(c2w))[None]
        sbsq, sb = probe_squared_footprint_weights(means, quats, scales, opac, vm, Ks,
                                                   width, height, n_probes, gen)
        vis = sb > 1e-6
        if not bool(vis.any()):
            continue
        idx = torch.nonzero(vis, as_tuple=True)[0]
        campos = t(np.asarray(c2w)[:3, 3])
        dirs = means[vis] - campos
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        phi = _sh_basis_torch(dirs, degree).float()
        w = sbsq[vis].float()
        D.index_add_(0, idx, w[:, None, None] * (phi[:, :, None] * phi[:, None, :]))
        w_tot.index_add_(0, idx, w)
        d_sum.index_add_(0, idx, w[:, None] * dirs)
    return D, w_tot, d_sum


def main(model, source, images_dir, iteration, n_probes, n_sample, n_dirs, device="cuda"):
    ck = read_3dgs_ply(str(Path(model) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"))
    for k in ("positions", "scales", "rotations", "opacities", "sh_coeffs"):
        ck[k] = np.asarray(ck[k], dtype=np.float32)
    degree = ck["sh_degree"]
    kdim = (degree + 1) ** 2

    orig = sorted((Path(model) / "renders" / "test" / "original").glob("*.npy"))
    gt0 = np.load(orig[0]) if orig else None
    if gt0 is not None:
        height, width = gt0.shape[1], gt0.shape[2]
    else:
        im = Image.open(next((Path(source) / images_dir).glob("*")))
        width, height = im.size
    test_cams, train_cams = load_cameras(Path(model), len(orig) if orig else 1)
    K_np = np.array(intrinsics(test_cams[0], width, height))
    frames = [(c["img_name"], c2w_opencv(c) @ _FLIP) for c in train_cams]
    print(f"{Path(model).name}: {ck['positions'].shape[0]:,} splats, {len(frames)} train views, "
          f"{width}x{height}, sh_degree {degree} (K={kdim})")

    D, w_tot, d_sum = accumulate_with_moments(ck, frames, K_np, width, height, degree,
                                              n_probes, device)
    lam = torch.tensor(empirical_band_precision(ck["sh_coeffs"], degree)[0],
                       dtype=torch.float32, device=device)
    sigma_n2 = 0.05 ** 2

    seen = torch.nonzero(w_tot > 0, as_tuple=True)[0]
    print(f"  {len(seen):,} splats observed ({100 * len(seen) / len(w_tot):.1f}%)")
    g = torch.Generator(device=device).manual_seed(0)
    pick = seen[torch.randperm(len(seen), generator=g, device=device)[:n_sample]]

    Ds = D[pick].double()
    ev = torch.linalg.eigvalsh(Ds)                       # ascending
    top_frac = (ev[:, -1] / ev.sum(dim=1).clamp_min(1e-30))
    top2_frac = (ev[:, -2:].sum(dim=1) / ev.sum(dim=1).clamp_min(1e-30))
    offdiag = (Ds - torch.diag_embed(torch.diagonal(Ds, dim1=-2, dim2=-1))).abs().sum((-1, -2))
    diagmass = torch.diagonal(Ds, dim1=-2, dim2=-1).abs().sum(-1)
    print(f"\n  spectrum of D_i over {len(pick):,} observed splats")
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<3} top-1 eigenvalue / trace = {np.percentile(top_frac.cpu(), q):.3f}"
              f"   top-2 = {np.percentile(top2_frac.cpu(), q):.3f}"
              f"   offdiag/diag = {np.percentile((offdiag / diagmass).cpu(), q):.3f}")

    # v_i(d) under each model, on random directions
    dirs = torch.randn((n_dirs, 3), generator=g, device=device, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    phi = _sh_basis_torch(dirs, degree).double()          # (n_dirs, K)
    Lam = torch.diag(lam.double())

    P_exact = Lam[None] + Ds / sigma_n2
    v_exact = torch.einsum("dk,nkl,dl->nd", phi, torch.linalg.inv(P_exact), phi)

    w = w_tot[pick].double()
    c_bd = float(1.0 / (4.0 * np.pi))
    P_bd = Lam[None] + (w[:, None, None] * c_bd) * torch.eye(kdim, dtype=torch.float64,
                                                             device=device)[None] / sigma_n2
    v_bd = torch.einsum("dk,nkl,dl->nd", phi, torch.linalg.inv(P_bd), phi)

    dbar = d_sum[pick].double()
    dbar = dbar / dbar.norm(dim=1, keepdim=True).clamp_min(1e-12)
    phib = _sh_basis_torch(dbar.float(), degree).double()  # (n_sample, K)
    P_r1 = Lam[None] + (w[:, None, None] / sigma_n2) * (phib[:, :, None] * phib[:, None, :])
    v_r1 = torch.einsum("dk,nkl,dl->nd", phi, torch.linalg.inv(P_r1), phi)

    def report(name, v):
        rel = ((v - v_exact).abs() / v_exact.clamp_min(1e-30)).flatten()
        a, b = v.flatten().cpu().numpy(), v_exact.flatten().cpu().numpy()
        r = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        print(f"    {name:16} median rel.err {np.percentile(rel.cpu(), 50):6.3f}   "
              f"p90 {np.percentile(rel.cpu(), 90):6.3f}   Spearman vs exact {r:.4f}")

    print(f"\n  v_i(d) against exact, over {n_sample:,} splats x {n_dirs} directions")
    report("band-diagonal", v_bd)
    report("rank-1", v_r1)
    print(f"\n  storage per splat: exact {kdim * (kdim + 1) // 2} floats, "
          f"rank-1 4, band-diagonal 1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-s", "--source", required=True)
    ap.add_argument("-i", "--images_dir", default="images")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--n_probes", type=int, default=32)
    ap.add_argument("--n_sample", type=int, default=20000)
    ap.add_argument("--n_dirs", type=int, default=16)
    a = ap.parse_args()
    main(a.model, a.source, a.images_dir, a.iteration, a.n_probes, a.n_sample, a.n_dirs)
