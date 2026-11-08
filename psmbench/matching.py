"""Алгоритми зіставлення. Кожен повертає MatchResult:
  pairs   — DataFrame(treated_idx, control_idx, weight, distance); weight — частка контролю у
            контрфактуалі treated_idx (для 1:1 → 1, для k:1 → 1/k, для kernel — нормовані ваги ядра)
  used_treated — булевий масив: чи має treated-об'єкт хоч одну пару (інакше випав через caliper/support)

Відстані для NN-зіставлення за схильністю — на логіт-шкалі (Austin 2011); caliper задається
в одиницях SD логіту. Ядерне зіставлення — на шкалі ймовірності (bandwidth у одиницях PS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class MatchResult:
    pairs: pd.DataFrame
    used_treated: np.ndarray
    n_treated: int
    n_control: int

    def control_weights(self, n_total: int) -> np.ndarray:
        """Сумарна вага кожного контрольного об'єкта (скільки разів і з якою вагою його використано)."""
        w = np.zeros(n_total)
        if len(self.pairs):
            np.add.at(w, self.pairs["control_idx"].values, self.pairs["weight"].values)
        return w


def _split(T: np.ndarray):
    return np.flatnonzero(T == 1), np.flatnonzero(T == 0)


def nn_match(score: np.ndarray, T: np.ndarray, k: int = 1, caliper: Optional[float] = None,
             replacement: bool = True, seed: int = 0) -> MatchResult:
    """k:1 найближчий сусід за одновимірною оцінкою `score` (логіт схильності).

    caliper — абсолютний поріг на тій самій шкалі (вже помножений на SD логіту).
    Без повторного вибору: жадібно, у випадковому порядку treated (seed), кожен контроль — один раз.
    """
    tr, co = _split(T)
    rng = np.random.default_rng(seed)
    co = rng.permutation(co)  # випадковий порядок контролю: рівні відстані (ties) розриваються випадково
    rows = []
    used = np.zeros(len(score), dtype=bool)
    if replacement:
        nn = NearestNeighbors(n_neighbors=min(k, len(co)))
        nn.fit(score[co].reshape(-1, 1))
        dist, ind = nn.kneighbors(score[tr].reshape(-1, 1))
        for p, i in enumerate(tr):
            ok = np.ones(dist.shape[1], dtype=bool) if caliper is None else dist[p] <= caliper
            if not ok.any():
                continue
            m = ok.sum()
            for j_pos, d in zip(ind[p][ok], dist[p][ok]):
                rows.append((i, co[j_pos], 1.0 / m, float(d)))
            used[i] = True
    else:
        order = rng.permutation(tr)
        avail = np.ones(len(co), dtype=bool)
        co_score = score[co]
        for i in order:
            if not avail.any():
                break
            d = np.abs(co_score - score[i])
            d = np.where(avail, d, np.inf)
            cand = np.argsort(d)[:k]
            cand = [c for c in cand if np.isfinite(d[c]) and (caliper is None or d[c] <= caliper)]
            if not cand:
                continue
            for c in cand:
                rows.append((i, co[c], 1.0 / len(cand), float(d[c])))
                avail[c] = False
            used[i] = True
    pairs = pd.DataFrame(rows, columns=["treated_idx", "control_idx", "weight", "distance"])
    return MatchResult(pairs=pairs, used_treated=used, n_treated=len(tr), n_control=len(co))


def kernel_match(e: np.ndarray, T: np.ndarray, bandwidth: float = 0.06, kernel: str = "epanechnikov",
                 common_support: bool = True) -> MatchResult:
    """Ядерне зіставлення на шкалі ймовірності: кожен treated отримує всі контролі з вагами K((e_j−e_i)/h)."""
    tr, co = _split(T)
    used = np.zeros(len(e), dtype=bool)
    lo, hi = e[co].min(), e[co].max()
    rows = []
    ec = e[co]
    for i in tr:
        if common_support and not (lo <= e[i] <= hi):
            continue
        u = (ec - e[i]) / bandwidth
        if kernel == "gaussian":
            kw = np.exp(-0.5 * u ** 2)
        elif kernel == "epanechnikov":
            kw = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
        else:
            raise ValueError(kernel)
        s = kw.sum()
        if s <= 0:
            continue
        kw = kw / s
        nz = np.flatnonzero(kw > 1e-10)
        for j in nz:
            rows.append((i, co[j], float(kw[j]), float(abs(ec[j] - e[i]))))
        used[i] = True
    pairs = pd.DataFrame(rows, columns=["treated_idx", "control_idx", "weight", "distance"])
    return MatchResult(pairs=pairs, used_treated=used, n_treated=len(tr), n_control=len(co))


def mahalanobis_match(X: np.ndarray, T: np.ndarray, k: int = 1, replacement: bool = True,
                      seed: int = 0, caliper: Optional[float] = None) -> MatchResult:
    """1:1 (k:1) найближчий сусід за відстанню Махаланобіса у просторі коваріат (без схильності)."""
    tr, co = _split(T)
    cov = np.cov(X, rowvar=False)
    cov = cov + 1e-8 * np.eye(cov.shape[0])
    VI = np.linalg.pinv(cov)
    used = np.zeros(len(T), dtype=bool)
    rows = []
    if replacement:
        nn = NearestNeighbors(n_neighbors=min(k, len(co)), metric="mahalanobis",
                              metric_params={"VI": VI}, algorithm="brute")
        nn.fit(X[co])
        dist, ind = nn.kneighbors(X[tr])
        for p, i in enumerate(tr):
            ok = np.ones(dist.shape[1], dtype=bool) if caliper is None else dist[p] <= caliper
            if not ok.any():
                continue
            m = ok.sum()
            for j_pos, d in zip(ind[p][ok], dist[p][ok]):
                rows.append((i, co[j_pos], 1.0 / m, float(d)))
            used[i] = True
    else:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(VI)
        Z = X @ L  # евклідова відстань у Z = Махаланобіс у X
        avail = np.ones(len(co), dtype=bool)
        for i in rng.permutation(tr):
            if not avail.any():
                break
            d = np.linalg.norm(Z[co] - Z[i], axis=1)
            d = np.where(avail, d, np.inf)
            cand = [c for c in np.argsort(d)[:k] if np.isfinite(d[c]) and (caliper is None or d[c] <= caliper)]
            if not cand:
                continue
            for c in cand:
                rows.append((i, co[c], 1.0 / len(cand), float(d[c])))
                avail[c] = False
            used[i] = True
    pairs = pd.DataFrame(rows, columns=["treated_idx", "control_idx", "weight", "distance"])
    return MatchResult(pairs=pairs, used_treated=used, n_treated=len(tr), n_control=len(co))
