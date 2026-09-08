# BQ-Splat RA-L draft

This folder contains an anonymous first-submission draft for *IEEE Robotics
and Automation Letters* (RA-L). It uses the official `ieeeconf.cls` initial-
submission template, US Letter paper, 10 pt, and two columns.

Build it with:

```bash
latexmk -pdf main.tex
```

Clean generated files with:

```bash
latexmk -C main.tex
```

The copied figures make the folder self-contained. Their source versions live
under `../gs_experiment/results/`. Numerical claims in the text come from
`../README.md`, `../bq_splat/results/FINDINGS.md`, and
`../gs_experiment/results/FINDINGS.md`.

## Draft status

The paper is intentionally conservative about the evidence currently in the
repository. In particular, it distinguishes:

- the exact alpha-compositing result on a piecewise-constant ray model;
- the standard RKHS worst-case-error result for kernel quadrature; and
- the practical 3-D local-window estimator, which is related but is not yet
  proved to bound rendered pixel error.

Before submission, the draft still needs:

- author and affiliation decisions (the review copy must remain anonymous);
- a dense, high-quality real-capture directional experiment;
- matched implementations of the principal uncertainty baselines;
- a runtime and memory table for the actual inference implementation;
- final confirmation of every result from archived machine-readable outputs;
- a pass against the current RA-L keyword list and PaperCept PDF checker; and
- final page balancing and copy-editing. The current build is six pages;
  RA-L permits at most eight pages, with charges for pages seven and eight.

The class file in this folder was downloaded from the RA-L/PaperCept template
link. Accepted papers use a different IEEE journal layout; do not switch this
review draft to that format before acceptance.
