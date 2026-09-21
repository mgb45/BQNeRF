# Patches applied to third-party checkouts

`third_party/` is gitignored (it holds upstream clones), so any local change
is recorded here instead, with the reason and the evidence that it is safe.

## `gsu-save_ply-memory.patch` — U-3DGS / 3DGS `scene/gaussian_model.py`

Applies to `github.com/Chumsy0725/GS-U` (and the original 3DGS, which has the
same code). Replaces

    elements[:] = list(map(tuple, attributes))

with a column-wise assignment in `save_ply` and `save_ply_uncertainty`.

**Why.** That line materialises one Python tuple per splat, each holding one
Python float per attribute: ~11 GB of interpreter objects for a 5.5M-splat
Mip-NeRF 360 outdoor scene, on top of the numpy array already holding the
same values. It OOM-killed `garden` at checkpoint save after 32 minutes of
completed training, three times, including once under a 13 GB cgroup cap
(journal: `gs-garden.service: Failed with result 'oom-kill'`).

**Why it is safe.** This is serialisation only — no model, training step, or
reported number depends on it. Verified byte-identical: building the
structured array both ways on a real 200k-splat checkpoint gives identical
raw bytes, and the serialised `.ply` files have the same sha256
(`58e873537ee0b063479cbb207366f1b8...`).

Nothing about the *method* comparison is affected; every U-3DGS number still
comes from their unmodified training, fitting and scoring code.
