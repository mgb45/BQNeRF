"""Project-page video assets.

Two assets per scene, both chosen because they need no colorbar, legend or
caption to land:

**flicker** -- one fixed camera, cycling posterior draws. Regions the training
views pinned down are pixel-steady; regions they did not visibly shimmer.
This is the asset no competing method can produce: our output is a set of
RENDERS, not a scalar field, so uncertainty can be shown as disagreement
between plausible scenes rather than as a heatmap a reader must be taught to
read. It is deliberately a single panel with no colormap.

**flythrough** -- the held-out views in angular order, five panels: ground
truth, the real render, our sigma, the residual-supervised baseline's sigma,
and the actual error. Our sigma and the error light up together; the
baseline's does not, which is FINDINGS section 17's -0.84 per-view
anti-correlation made visible rather than tabulated.

Both are built on the `gap75` checkpoints (a 75 degree cone of training views
removed, retrained at full capacity). That is a deliberate choice, not a
flattering one: FINDINGS section 4 showed that on a fully-observed 100-view
checkpoint the posterior draws are pixel-identical, which is the correct
answer and a useless animation. An honest hero has to be a regime where the
method has something to say.

Run: .venv-gsplat/bin/python gs_experiment/scripts/make_project_page_videos.py [scene ...]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from gs_experiment.baselines import fit_residual_supervised_sh, render_sh_field
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_geometry_ensemble import render
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR, SCENES

VIDEO_DIR = RESULTS_DIR / "project_page"
N_DRAWS = 16
# Half the original 8 fps. At 8 the flicker read as noise and the flythrough
# moved past each held-out view before the third and fifth panels could be
# compared -- the whole point of putting them side by side.
FLICKER_FPS = 4
FLYTHROUGH_FPS = 4
# The single-panel hero is upscaled; the five-panel strip is already ~2000px
# wide at native resolution, and upscaling it only quadrupled file size.
FLICKER_UPSCALE = 2
PANEL_UPSCALE = 1
CRF_FLICKER, CRF_PANEL = 18, 23
LABELS = ["ground truth", "render", "ours", "residual-supervised", "|error|"]


def _encode(frame_dir: Path, out_stem: Path, fps: int, crf: int = 18):
    """H.264 for broad autoplay support plus VP9, and a poster frame. yuv420p
    and even dimensions are required by Safari/iOS, which silently refuse
    odd-sized h264."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    common = ["-y", "-loglevel", "error", "-framerate", str(fps),
              "-i", str(frame_dir / "f%05d.png"),
              "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    subprocess.run(["ffmpeg", *common, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(crf), "-movflags", "+faststart", f"{out_stem}.mp4"], check=True)
    subprocess.run(["ffmpeg", *common, "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
                    "-crf", str(crf + 14), "-b:v", "0", f"{out_stem}.webm"], check=True)
    shutil.copy(frame_dir / "f00000.png", f"{out_stem}_poster.png")


def _save(frames: list[np.ndarray], stem: Path, fps: int, pingpong=True,
          upscale: int = 1, crf: int = 18):
    """Write frames and encode. Ping-pong (forward then back, endpoints not
    repeated) makes a seamless loop out of a non-cyclic sequence, so the page
    never shows a jump-cut."""
    seq = frames + frames[-2:0:-1] if pingpong and len(frames) > 2 else frames
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i, f in enumerate(seq):
            img = Image.fromarray(np.clip(f * 255, 0, 255).astype(np.uint8))
            if upscale != 1:
                img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
            img.save(d / f"f{i:05d}.png")
        _encode(d, stem, fps, crf)
    print(f"  wrote {stem}.mp4 / .webm ({len(seq)} frames @ {fps}fps)")


def _heat(x: np.ndarray, vmax: float) -> np.ndarray:
    return cm.inferno(np.clip(x / max(vmax, 1e-12), 0, 1))[..., :3]


def _label_strip(width: int, panels: int) -> np.ndarray:
    """Panel captions burned in, so the video is self-describing if it is
    ever shown without its surrounding page."""
    from PIL import ImageDraw, ImageFont

    scale = PANEL_UPSCALE
    font_px = max(14, int(0.035 * (width // panels) * scale))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_px)
    except OSError:
        font = ImageFont.load_default()
    h = int(font_px * 2.0)
    img = Image.new("RGB", (width * scale, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw = (width // panels) * scale
    for i, name in enumerate(LABELS[:panels]):
        draw.text((i * pw + pw // 2, h // 2), name, fill=(20, 20, 20), font=font, anchor="mm")
    return np.asarray(img, dtype=np.float32) / 255.0


def scene_assets(scene: str, ckpt_name: str, eval_name: str = "eval"):
    ckpt_dir = LOCAL_RUNS / f"{scene}_prepared" / ckpt_name
    eval_dir = LOCAL_RUNS / f"{scene}_prepared" / eval_name
    if not (ckpt_dir / "splats.ply").exists():
        print(f"{scene}: no {ckpt_name} checkpoint, skipping")
        return
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]
    cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        tw, th = im.size
    train_K = fov_x_to_intrinsics(cax, tw, th)
    print(f"{scene}/{ckpt_name}: {sh_coeffs.shape[0]} splats, {len(train_frames)} training views")

    sigma_n = float(np.sqrt(estimate_noise_variance(
        checkpoint, train_frames, train_K, tw, th, str(ckpt_dir),
        background_color=BACKGROUND_COLOR)))
    data = accumulate_sh_precision_rasterized(
        checkpoint, train_frames, train_K, tw, th, degree, n_probes=N_PROBES,
        seed=SEED, device="cuda", progress_every=0) / (sigma_n ** 2)
    draws = sample_sh_draws(sh_coeffs, data, empirical_band_precision(sh_coeffs, degree),
                            N_DRAWS, SEED)
    psi = fit_residual_supervised_sh(checkpoint, train_frames, train_K, tw, th, degree,
                                     str(ckpt_dir), lam=1.0, n_cg_iters=120,
                                     background_color=BACKGROUND_COLOR, verbose=False)

    ecax, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    with Image.open(str(eval_dir / (eval_frames[0][0] + ".png"))) as im:
        w, h = im.size
    K = fov_x_to_intrinsics(ecax, w, h)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")  # noqa: E731
    bg, theta_hat, op_hat = t(BACKGROUND_COLOR), t(sh_coeffs), t(checkpoint["opacities"])

    # Order held-out views by angle from the removed cone's centre, so the
    # flythrough walks from best-covered to least-covered rather than in
    # arbitrary capture order.
    tc = np.array([np.asarray(c2w)[:3, 3] for _, c2w in train_frames])
    td_ = tc / np.linalg.norm(tc, axis=1, keepdims=True)
    ec = np.array([np.asarray(c2w)[:3, 3] for _, c2w in eval_frames])
    ed = ec / np.linalg.norm(ec, axis=1, keepdims=True)
    gap_dir = -td_.mean(axis=0)
    gap_dir /= np.linalg.norm(gap_dir)
    order = np.argsort(ed @ gap_dir)[::-1]          # most-inside-the-gap first

    per_view = []
    for vi in order:
        file_path, c2w = eval_frames[vi]
        viewmat, Ks = t(opencv_viewmat_from_c2w(c2w))[None], t(K)[None]
        gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"),
                        dtype=np.float32) / 255.0
        mean_render = render(checkpoint, theta_hat, op_hat, viewmat, Ks, w, h, bg)
        ens = np.stack([render(checkpoint, draws[s], op_hat, viewmat, Ks, w, h, bg)
                        for s in range(N_DRAWS)], axis=0)
        per_view.append({
            "gt": gt, "render": mean_render, "ens": ens,
            "ours": ens.std(axis=0).mean(axis=2),
            "rs": np.abs(render_sh_field(checkpoint, psi, c2w, K, w, h, degree)).mean(axis=2),
            "err": np.abs(mean_render - gt).mean(axis=2)})

    # Fixed normalisation across the whole sequence, per field: a per-frame
    # vmax would make brightness pulse with the colormap rescaling rather
    # than with the uncertainty, which is exactly the wrong signal.
    vmax = {k: float(np.percentile(np.stack([p[k] for p in per_view]), 99.5))
            for k in ("ours", "rs", "err")}

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    hero = per_view[0]                              # deepest into the removed cone
    _save([hero["ens"][s] for s in range(N_DRAWS)],
          VIDEO_DIR / f"{scene}_flicker", FLICKER_FPS,
          upscale=FLICKER_UPSCALE, crf=CRF_FLICKER)

    strip = _label_strip(w * 5, 5)
    frames = []
    for p in per_view:
        row = np.concatenate([p["gt"], p["render"], _heat(p["ours"], vmax["ours"]),
                              _heat(p["rs"], vmax["rs"]), _heat(p["err"], vmax["err"])], axis=1)
        if PANEL_UPSCALE != 1:
            row = np.asarray(Image.fromarray((row * 255).astype(np.uint8)).resize(
                (row.shape[1] * PANEL_UPSCALE, row.shape[0] * PANEL_UPSCALE), Image.LANCZOS),
                dtype=np.float32) / 255.0
        frames.append(np.concatenate([strip, row], axis=0))
    with tempfile.TemporaryDirectory() as td2:
        d = Path(td2)
        seq = frames + frames[-2:0:-1]
        for i, f in enumerate(seq):
            Image.fromarray(np.clip(f * 255, 0, 255).astype(np.uint8)).save(d / f"f{i:05d}.png")
        _encode(d, VIDEO_DIR / f"{scene}_flythrough", FLYTHROUGH_FPS, CRF_PANEL)
    print(f"  wrote {VIDEO_DIR / f'{scene}_flythrough'}.mp4 / .webm ({len(seq)} frames)")


def run(scenes=None):
    for scene in (scenes or SCENES):
        ckpt = "gap75" if (LOCAL_RUNS / f"{scene}_prepared" / "gap75" / "splats.ply").exists() else "wide"
        scene_assets(scene, ckpt)
    if (LOCAL_RUNS / "bonsai_prepared" / "gap_0" / "splats.ply").exists():
        scene_assets("bonsai", "gap_0")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
