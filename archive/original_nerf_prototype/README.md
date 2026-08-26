# The original prototype (archived)

This is where the project started: a small, from-scratch NeRF
(`models/nerf.py`, trained by `train.py`) that replaces standard
volume-rendering quadrature with Bayesian quadrature under an RBF kernel,
giving a closed-form posterior variance used directly in a Gaussian
negative-log-likelihood training loss. `tutorial/Intro to Bayesian
quadrature.ipynb` is a from-scratch derivation of the method; `inspect_model.ipynb`
loads a trained checkpoint and reproduces the `figs/Ns_*.png` PSNR-vs-
sample-count comparison.

**Why it's archived rather than developed further**: the one experiment
here (PSNR vs. sample count, BQ vs. standard quadrature, on a single
held-out image) showed BQ matching or losing to standard quadrature at
every sample count — and by the time this was revisited, 3D Gaussian
Splatting had made pure ray-marching NeRF pipelines largely uncompetitive
on the metric this prototype was implicitly judged on. A literature check
at that point found the real opening wasn't here: it was applying the
same Bayesian-quadrature idea to Gaussian Splatting instead, where it
turns out to unify two things the literature treats separately
(quadrature/discretization uncertainty and directional/epistemic
uncertainty) — see the top-level `README.md` and `ROADMAP.md` for that
whole story, which is what the rest of this repository is about.

Nothing here is imported by the current codebase (`bq_splat/`,
`gs_experiment/`) — it's kept only as the origin record. The one
substantive idea that *did* carry forward, the closed-form kernel-mean-
embedding derivation (`rbf_vf`/`rbf_vff`/`rbf_vvf_part` in
`models/nerf.py`), was independently re-derived and validated from
scratch in `bq_splat/kernels.py`, cross-checked against this file's exact
formula in `tests/test_kernels.py` — not imported from here.

## Running it (as it was)

```
python3 train.py --bq BQ --nsamples 64 --lr 5e-4 --epochs 5000    # Bayesian quadrature
python3 train.py --bq Std --nsamples 64 --lr 5e-4 --epochs 5000   # standard quadrature baseline
```

Logs images, videos, and checkpoints during training; `inspect_model.ipynb`
loads a trained checkpoint for further inspection.
