"""Propensity-score models e(X) = P(T=1 | X).

model:
  'lr'     — logistic regression, main effects only (as in the thesis), practically unregularised (C=1e6)
  'lr_dw'  — Dehejia & Wahba (1999) specification: + squares of continuous covariates, + u74, u75 (LaLonde only)
  'gbm'    — HistGradientBoostingClassifier (depth 3, lr 0.05, early stopping), out-of-fold predictions
  'rf'     — RandomForest (500 trees, min_samples_leaf 20): OOB probabilities + isotonic calibration

For the ML models e(X) is predicted out-of-fold (5 folds) so that the propensity is not overfitted
on the training data (an overfitted e(X) ≈ T destroys overlap). LR is predicted in-sample,
as in classical PSM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


@dataclass
class PropensityResult:
    e: np.ndarray            # P(T=1|X), clipped to [EPS, 1-EPS]
    logit: np.ndarray
    metrics: Dict[str, float]
    model: str


def design_matrix(df: pd.DataFrame, spec, model: str) -> pd.DataFrame:
    """Feature matrix for the propensity model (extended for lr_dw)."""
    X = df[spec.covariates].astype(float).copy()
    if model == "lr_dw":
        for c in spec.squared_terms:
            X[f"{c}_sq"] = X[c] ** 2
        for c in spec.extra_ps_covariates:
            if c in df.columns:
                X[c] = df[c].astype(float)
    return X


def _make_estimator(model: str, seed: int):
    if model in ("lr", "lr_dw"):
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs"))
    if model == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=1000,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=30,
            l2_regularization=1.0, random_state=seed)
    if model == "rf":
        # OOB prediction instead of k-fold: each unit is scored only by the trees that did not see it;
        # isotonic calibration on the OOB probabilities (RF probabilities are shrunk towards 0.5)
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, oob_score=True,
                                      n_jobs=1, random_state=seed)
    raise ValueError(f"unknown propensity model: {model}")


def fit_propensity(df: pd.DataFrame, spec, model: str = "lr", seed: int = 0,
                   oof_folds: int = 5) -> PropensityResult:
    X = design_matrix(df, spec, model).values
    T = df[spec.treatment].astype(int).values
    est = _make_estimator(model, seed)

    if model in ("lr", "lr_dw"):
        est.fit(X, T)
        p = est.predict_proba(X)[:, 1]
    elif model == "rf":
        est.fit(X, T)
        oob = est.oob_decision_function_[:, 1]
        oob = np.where(np.isnan(oob), T.mean(), oob)  # a unit that fell into every bootstrap sample
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oob, T)
        p = iso.predict(oob)
    else:
        p = np.empty(len(T))
        skf = StratifiedKFold(n_splits=oof_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, T):
            m = clone(est)
            m.fit(X[tr], T[tr])
            p[te] = m.predict_proba(X[te])[:, 1]

    e = np.clip(p, EPS, 1 - EPS)
    metrics = {
        "auc": float(roc_auc_score(T, e)),
        "logloss": float(log_loss(T, e)),
        "brier": float(brier_score_loss(T, e)),
        "frac_e_gt_095": float(np.mean(e > 0.95)),
    }
    return PropensityResult(e=e, logit=np.log(e / (1 - e)), metrics=metrics, model=model)
