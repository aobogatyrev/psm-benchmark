"""Стандартні похибки: Abadie–Imbens (2006) для NN-зіставлення та бутстреп."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _within_group_sigma2(Y: np.ndarray, score: np.ndarray, T: np.ndarray, J: int = 1) -> np.ndarray:
    """Оцінка умовної дисперсії σ²(X_i, T_i) через J найближчих сусідів тієї самої групи
    (Abadie & Imbens 2006, розд. 6): σ̂²_i = J/(J+1) · (Y_i − mean(Y сусідів))²."""
    s2 = np.full(len(Y), np.nan)
    for g in (0, 1):
        idx = np.flatnonzero(T == g)
        if len(idx) <= J:
            s2[idx] = np.var(Y[idx], ddof=1) if len(idx) > 1 else 0.0
            continue
        nn = NearestNeighbors(n_neighbors=J + 1).fit(score[idx].reshape(-1, 1))
        _, ind = nn.kneighbors(score[idx].reshape(-1, 1))
        nb = Y[idx][ind[:, 1:]].mean(axis=1)  # сусіди без самого себе
        s2[idx] = J / (J + 1) * (Y[idx] - nb) ** 2
    return s2


def abadie_imbens_se(Y: np.ndarray, T: np.ndarray, score: np.ndarray, w: np.ndarray,
                     used_treated: np.ndarray, J: int = 1) -> float:
    """SE оцінки ATT для зіставлення з повторним вибором (умовна на X та зіставлення дисперсія):
        V = (1/N1u²) [ Σ_{i treated, used} σ̂²_i + Σ_{j control} w_j² σ̂²_j ],
    де w_j — сумарна вага контролю j (K_M(j)/M в позначеннях AI), N1u — кількість використаних treated.
    Це «sample ATT» варіант без доданка на неоднорідність ефекту (Abadie & Imbens 2006, Thm 7;
    те, що за замовчуванням дає R::Matching з Var.calc=0 у гомоскедастичному випадку, тут —
    з локальною оцінкою дисперсії через J сусідів)."""
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
    """Непараметричний бутстреп зі стратифікацією за T; point_fn(df_boot, seed) → ATT
    (переоцінює модель схильності та зіставлення/ваги у кожній вибірці).
    Повертає (SE, перцентильний 95 % CI, масив оцінок)."""
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(df[treatment].values == 1)
    co = np.flatnonzero(df[treatment].values == 0)
    ests = np.full(B, np.nan)
    for b in range(B):
        idx = np.concatenate([rng.choice(tr, len(tr), replace=True), rng.choice(co, len(co), replace=True)])
        boot = df.iloc[idx].reset_index(drop=True)
        try:
            ests[b] = point_fn(boot, seed + 1000 + b)
        except Exception:  # вироджена вибірка (наприклад, жодної пари) — пропускаємо
            ests[b] = np.nan
    ok = ests[np.isfinite(ests)]
    if len(ok) < 10:
        return np.nan, (np.nan, np.nan), ests
    return float(ok.std(ddof=1)), (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))), ests
