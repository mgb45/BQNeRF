"""Does the per-pixel calibration transfer across scenes?

FINDINGS section 9 established that the raw posterior is underconfident by a
near-constant factor, and that a two-parameter fit
`sigma_total^2 = s^2 sigma_pred^2 + sigma_0^2` makes it properly calibrated.
Those two numbers were fitted PER CHECKPOINT on held-out views. They came out
close on two very different lego checkpoints (s = 5.5 and 6.4), which hints
they might be a property of the construction rather than of a scene -- but a
hint from two checkpoints of the same object is not evidence.

This is the difference between "calibrated given a validation split of the
scene you are working on" and "calibrated out of the box", so it is worth
settling. Over all 7 NeRF-Synthetic scenes, `wide` checkpoints, 30 held-out
eval views each:

  * fit (s, sigma_0) per scene, and look at the spread;
  * LEAVE-ONE-SCENE-OUT: fit on the other six scenes' views, score the
    held-out scene, and compare against both a constant-variance baseline
    and that scene's own fitted calibration (the achievable target).

If leave-one-scene-out lands close to the per-scene fit, the constants
transfer and can be reported as constants of the method.

Run: .venv-gsplat/bin/python gs_experiment/scripts/analyse_calibration_transfer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gs_experiment.scripts.analyse_pixel_calibration import _nll, collect, fit_variance_model
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR, SCENES

N_DRAWS = 16
SUBSAMPLE = 200_000   # pixel-channels kept per scene split; the fits are far past converged here


def run():
    rng = np.random.default_rng(0)
    data = {}
    for scene in SCENES:
        groups, sigma_n = collect(LOCAL_RUNS / f"{scene}_prepared", "wide", "eval", None, n_draws=N_DRAWS)
        fit_idx = list(range(0, len(groups), 2))
        test_idx = [i for i in range(len(groups)) if i not in fit_idx]

        def pool(idxs):
            sig = np.concatenate([groups[i][0] for i in idxs])
            res = np.concatenate([groups[i][1] for i in idxs])
            if len(sig) > SUBSAMPLE:
                k = rng.choice(len(sig), SUBSAMPLE, replace=False)
                sig, res = sig[k], res[k]
            return sig, res

        data[scene] = {"fit": pool(fit_idx), "test": pool(test_idx), "sigma_n": sigma_n}
        sf, rf = data[scene]["fit"]
        _, params = fit_variance_model(sf, rf, None)
        data[scene]["s"] = float(np.sqrt(np.exp(params[0])))
        data[scene]["s0"] = float(np.sqrt(np.exp(params[1])))
        print(f"{scene:<8} n={len(sf):>7}  s = {data[scene]['s']:6.3f}   sigma_0 = {data[scene]['s0']:.5f}"
              f"   training sigma_n = {sigma_n:.5f}   ratio sigma_0/sigma_n = {data[scene]['s0']/sigma_n:5.2f}")

    ss = np.array([data[s]["s"] for s in SCENES])
    s0s = np.array([data[s]["s0"] for s in SCENES])
    print(f"\nacross scenes: s mean {ss.mean():.3f} sd {ss.std():.3f} "
          f"(min {ss.min():.3f}, max {ss.max():.3f}, ratio max/min {ss.max()/ss.min():.2f})")
    print(f"               sigma_0 mean {s0s.mean():.5f} sd {s0s.std():.5f} "
          f"(ratio max/min {s0s.max()/s0s.min():.2f})")

    print("\nsigma_0 vs the scene's own training residual:")
    r_ = np.array([data[sc]["s0"] / data[sc]["sigma_n"] for sc in SCENES])
    print(f"  sigma_0/sigma_n across scenes: mean {r_.mean():.3f} sd {r_.std():.3f} "
          f"ratio max/min {r_.max()/r_.min():.2f}  (vs {s0s.max()/s0s.min():.2f} for raw sigma_0)")

    print(f"\n{'scene':<8}{'constant':>11}{'transferred':>13}{'anchored':>10}{'own fit':>10}")
    rows = []
    for scene in SCENES:
        st, rt = data[scene]["test"]
        others = [x for x in SCENES if x != scene]
        s_other = np.concatenate([data[o]["fit"][0] for o in others])
        r_other = np.concatenate([data[o]["fit"][1] for o in others])
        f_trans, _ = fit_variance_model(s_other, r_other, None)      # never sees this scene
        f_own, _ = fit_variance_model(*data[scene]["fit"], None)     # this scene's own fit views

        # Anchored: the floor is the OTHER scenes' fitted multiple of each
        # scene's own training residual, so nothing scene-specific is
        # transferred -- only two dimensionless constants.
        f_anch_feats = np.concatenate([np.full_like(data[o]["fit"][0], data[o]["sigma_n"] ** 2)
                                       for o in others])[:, None]
        f_anch, _ = fit_variance_model(s_other, r_other, f_anch_feats)
        test_feats = np.full_like(st, data[scene]["sigma_n"] ** 2)[:, None]

        nll_const = _nll(np.full_like(rt, np.var(data[scene]["fit"][1])), rt)
        nll_trans = _nll(f_trans(st, None), rt)
        nll_anch = _nll(f_anch(st, test_feats), rt)
        nll_own = _nll(f_own(st, None), rt)
        rows.append((scene, nll_const, nll_trans, nll_anch, nll_own))
        print(f"{scene:<8}{nll_const:>11.4f}{nll_trans:>13.4f}{nll_anch:>10.4f}{nll_own:>10.4f}")

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    names = ["constant", "transferred", "anchored", "own fit"]
    print("\nmean over scenes: " + "  ".join(f"{n} {arr[:, k].mean():.4f}" for k, n in enumerate(names)))
    for k, n in ((1, "transferred"), (2, "anchored")):
        print(f"  {n:<12} beats constant by {(arr[:, 0] - arr[:, k]).mean():+.4f} nats, "
              f"on {int((arr[:, k] < arr[:, 0]).sum())}/{len(SCENES)} scenes; "
              f"short of a per-scene fit by {(arr[:, k] - arr[:, 3]).mean():+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].bar(SCENES, ss, color="steelblue")
    axes[0].axhline(ss.mean(), ls="--", c="k", lw=1, label=f"mean {ss.mean():.2f}")
    axes[0].set_ylabel("fitted scale $s$"); axes[0].legend()
    axes[0].set_title("Underconfidence factor, fitted per scene", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    x = np.arange(len(SCENES))
    axes[1].bar(x - 0.3, arr[:, 0], 0.2, label="constant variance")
    axes[1].bar(x - 0.1, arr[:, 1], 0.2, label="transferred (leave-one-scene-out)")
    axes[1].bar(x + 0.1, arr[:, 2], 0.2, label="anchored to own $\\sigma_n$ (LOSO)")
    axes[1].bar(x + 0.3, arr[:, 3], 0.2, label="own fit")
    axes[1].set_xticks(x); axes[1].set_xticklabels(SCENES, rotation=45)
    axes[1].set_ylabel("Gaussian NLL on held-out views (lower better)"); axes[1].legend(fontsize=9)
    axes[1].set_title("Does the calibration transfer across scenes?", fontsize=12)
    fig.tight_layout()
    path = RESULTS_DIR / "calibration_transfer.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
