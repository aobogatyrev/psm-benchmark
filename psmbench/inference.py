"""Standard errors: Abadie–Imbens (2006) for NN matching and the bootstrap."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _within_group_sigma2(Y: np.ndarray, score: np.ndarray, T: np.ndarray, J: int = 1) -> np.ndarray:
    """Conditional variance σ²(X_i, T_i) estimated from the J nearest neighbours of the same arm
    (Abadie & Imbens 2006, Section 6): σ̂²_i = J/(J+1) · (Y_i − mean(Y of the neighbours))²."""
    s2 = np.full(len(Y), np.nan)
    for g in (0, 1):
        idx = np.flatnonzero(T == g)
        if len(idx) <= J:
            s2[idx] = np.var(Y[idx], ddof=1) if len(idx) > 1 else 0.0
            continue
        nn = NearestNeighbors(n_neighbors=J + 1).fit(score[idx].reshape(-1, 1))
        _, ind = nn.kneighbors(score[idx].reshape(-1, 1))
        nb = Y[idx][ind[:, 1:]].mean(axis=1)  # neighbours excluding the unit itself
        s2[idx] = J / (J + 1) * (Y[idx] - nb) ** 2
    return s2


def abadie_imbens_se(Y: np.ndarray, T: np.ndarray, score: np.ndarray, w: np.ndarray,
                     used_treated: np.ndarray, J: int = 1) -> float:
    """SE of the ATT estimate for matching with replacement (variance conditional on X and on the matching):
        V = (1/N1u²) [ Σ_{i treated, used} σ̂²_i + Σ_{j control} w_j² σ̂²_j ],
    where w_j is the total weight of control j (K_M(j)/M in the AI notation) and N1u the number of treated units used.
    This is the "sample ATT" variant without the effect-heterogeneity term (Abadie & Imbens 2006, Thm 7;
    what R::Matching returns by default with Var.calc=0 in the homoskedastic case; here with
    a local variance estimate from J neighbours)."""
    s2 = _within_group_sigma2(Y, score, T, J)
    tr = (T == 1) & used_treated
    co = T == 0
    n1 = tr.sum()
    if n1 == 0:
        return np.nan
    V = (s2[tr].sum() + ((w[co] ** 2) * s2[co]).sum()) / n1 ** 2
    return float(np.sqrt(V))


def stratified_bootstrap(df: pd.DataFrame, treatment: str, point_fn: Callable[[pd.DataFrame, int], float],
                         B: int = 500, seed: int = 0) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Non-parametric bootstrap stratified by T; point_fn(df_boot, seed) → ATT
    (the propensity model and the matching/weights are re-estimated in every resample).
    Returns (SE, percentile 95 % CI, array of estimates)."""
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(df[treatment].values == 1)
    co = np.flatnonzero(df[treatment].values == 0)
    ests = np.full(B, np.nan)
    for b in range(B):
        idx = np.concatenate([rng.choice(tr, len(tr), replace=True), rng.choice(co, len(co), replace=True)])
        boot = df.iloc[idx].reset_index(drop=True)
        try:
            ests[b] = point_fn(boot, seed + 1000 + b)
        except Exception:  # degenerate resample (e.g. no matched pair) — skipped
            ests[b] = np.nan
    ok = ests[np.isfinite(ests)]
    if len(ok) < 10:
        return np.nan, (np.nan, np.nan), ests
    return float(ok.std(ddof=1)), (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))), ests
