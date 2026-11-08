"""Covariate-balance diagnostics on the weighted sample."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _wmean(x, w):
    return float(np.average(x, weights=w)) if w.sum() > 0 else np.nan


def smd_table(df: pd.DataFrame, T: np.ndarray, w: np.ndarray, covariates: List[str],
              continuous: List[str]) -> pd.DataFrame:
    """SMD before (uniform weights) and after (weights w) for every covariate.

    The denominator is the pooled SD BEFORE balancing (cobalt convention), the same for "before" and "after",
    so that changes in SMD reflect changes in means, not in variances. For binary covariates — raw difference in proportions.
    """
    tr, co = T == 1, T == 0
    rows = []
    for c in covariates:
        x = df[c].astype(float).values
        if c in continuous:
            sd = np.sqrt((x[tr].var(ddof=1) + x[co].var(ddof=1)) / 2)
        else:
            sd = 1.0
        before = (x[tr].mean() - x[co].mean()) / sd
        after = (_wmean(x[tr], w[tr]) - _wmean(x[co], w[co])) / sd
        rows.append({"covariate": c, "smd_before": before, "smd_after": after})
    return pd.DataFrame(rows)


def balance_summary(tab: pd.DataFrame) -> Dict[str, float]:
    a = tab["smd_after"].abs()
    b = tab["smd_before"].abs()
    return {
        "mean_abs_smd_before": float(b.mean()),
        "max_abs_smd_before": float(b.max()),
        "mean_abs_smd_after": float(a.mean()),
        "max_abs_smd_after": float(a.max()),
        "frac_smd_lt_01_after": float((a < 0.1).mean()),
    }


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0
