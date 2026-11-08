"""Weighting: IPW for the ATT and entropy balancing (Hainmueller 2012)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


def ipw_att_weights(e: np.ndarray, T: np.ndarray, trim: Optional[float] = 0.01) -> np.ndarray:
    """ATT weights: treated → 1, control → e/(1−e). Normalisation (Hájek) is done in estimators.

    trim: controls with e > 1−trim (almost surely "treated" by their covariates, extreme weights)
    get weight 0; treated units outside the common support of the controls are excluded as well.
    """
    w = np.where(T == 1, 1.0, e / (1 - e))
    if trim is not None and trim > 0:
        w = np.where((T == 0) & (e > 1 - trim), 0.0, w)
        co = T == 0
        lo, hi = e[co & (w > 0)].min(), e[co & (w > 0)].max()
        w = np.where((T == 1) & ((e < lo) | (e > hi)), 0.0, w)
    return w


def _moments(X: np.ndarray, continuous_mask: np.ndarray) -> np.ndarray:
    """First moments of all covariates + second moments of the continuous ones."""
    cols = [X]
    if continuous_mask.any():
        cols.append(X[:, continuous_mask] ** 2)
    M = np.hstack(cols)
    # drop constant and duplicated columns (singularity)
    keep = M.std(axis=0) > 1e-12
    M = M[:, keep]
    _, idx = np.unique(np.round(M.T, 10), axis=0, return_index=True)
    return M[:, np.sort(idx)]


def entropy_balancing(X: np.ndarray, T: np.ndarray, continuous_mask: np.ndarray,
                      max_iter: int = 500) -> Tuple[np.ndarray, dict]:
    """Control weights that reproduce the treated moments exactly (1st for all, 2nd for continuous covariates)
    while staying closest to uniform weights in the entropy sense. Dual problem (convex):
        min_λ  log Σ_j exp(−λ'(c_j − c̄_T)),   w_j ∝ exp(−λ'(c_j − c̄_T)).
    Returns weights over all units (treated → 1, control → w_j, Σ_j w_j = 1).
    """
    tr, co = T == 1, T == 0
    M = _moments(X, continuous_mask)
    target = M[tr].mean(axis=0)
    C = M[co] - target
    # scale the columns for conditioning
    scale = C.std(axis=0)
    scale[scale == 0] = 1.0
    Cs = C / scale

    def objective(lam):
        z = -Cs @ lam
        zmax = z.max()
        lse = zmax + np.log(np.exp(z - zmax).sum())
        w = np.exp(z - lse)
        grad = -(Cs * w[:, None]).sum(axis=0)
        return lse, grad

    res = minimize(objective, np.zeros(Cs.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-9})
    z = -Cs @ res.x
    w_c = np.exp(z - z.max())
    w_c /= w_c.sum()
    w = np.zeros(len(T))
    w[tr] = 1.0
    w[co] = w_c
    # residual moment imbalance in standardised units (SD of the moment among the controls)
    max_imbalance = float((np.abs((M[co] * w_c[:, None]).sum(axis=0) - target) / scale).max())
    info = {"converged": bool(res.success), "n_moments": int(Cs.shape[1]),
            "max_moment_imbalance": max_imbalance}
    return w, info
