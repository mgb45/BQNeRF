"""A simple visibility-coverage proxy, standing in for a real visibility
field (GAVIS-style) or Hessian/Fisher sensitivity (PUP-style) where full
reproduction of either is out of scope (see ROADMAP.md's "Downstream
evaluation, as combination rather than competition"). Deliberately not
Bayesian-quadrature-based -- the point of the differentiation experiment is
comparing BQ's directional-kernel signal against a genuinely different
mechanism, not another flavor of the same one.

Uses standard circular/spherical statistics (the mean resultant length),
not anything from bq_splat.
"""

from __future__ import annotations

import numpy as np


def resultant_length(directions: np.ndarray) -> float:
    """Mean resultant length R of a set of unit direction vectors: the norm
    of their average. R close to 1 means the directions are tightly
    clustered (poor angular coverage); R close to 0 means they point every
    which way (good angular coverage). Standard directional-statistics
    quantity (see e.g. Mardia & Jupp, "Directional Statistics"), computed
    here with no reference to any kernel or GP.
    """
    directions = np.asarray(directions, dtype=float)
    if directions.shape[0] == 0:
        return 1.0  # no observations at all -- treat as maximally poor coverage
    mean_vec = directions.mean(axis=0)
    return float(np.linalg.norm(mean_vec))


def visibility_uncertainty_proxy(directions: np.ndarray, min_count: int = 3, sparse_penalty: float = 1.0) -> float:
    """A single scalar "how untrustworthy is this region" proxy: high when
    either (a) there are too few observations to say anything (below
    `min_count`), or (b) the observations that exist are angularly
    concentrated (high resultant length). Not calibrated to be on the same
    scale as BQ posterior variance -- the differentiation experiment
    compares whether the two signals rank regions the same way, not their
    absolute values.
    """
    directions = np.asarray(directions, dtype=float)
    n = directions.shape[0]
    if n < min_count:
        return sparse_penalty
    return resultant_length(directions)
