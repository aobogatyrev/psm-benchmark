"""Моделі оцінки схильності e(X) = P(T=1 | X).

model:
  'lr'     — логістична регресія без взаємодій (як у КР), практично без регуляризації (C=1e6)
  'lr_dw'  — специфікація Dehejia & Wahba 1999: + квадрати неперервних, + u74, u75 (лише LaLonde)
  'gbm'    — HistGradientBoostingClassifier (depth 3, lr 0.05, early stopping), out-of-fold прогноз
  'rf'     — RandomForest (500 дерев, min_samples_leaf 20): OOB-ймовірності + ізотонічне калібрування

Для ML-моделей e(X) прогнозується out-of-fold (5-fold), щоб не переоцінювати схильність
на навчальних даних (перенавчена e(X) ≈ T руйнує перекриття). LR прогнозується in-sample,
як у класичному PSM.
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
    e: np.ndarray            # P(T=1|X), обрізано до [EPS, 1-EPS]
    logit: np.ndarray
    metrics: Dict[str, float]
    model: str


def design_matrix(df: pd.DataFrame, spec, model: str) -> pd.DataFrame:
    """Матриця ознак для моделі схильності (для lr_dw — розширена)."""
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
        # OOB-прогноз замість k-fold: кожен об'єкт оцінюють лише дерева, що його не бачили;
        # калібрування ізотонічною регресією на OOB-ймовірностях (RF-ймовірності зміщені до 0.5)
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, oob_score=True,
                                      n_jobs=1, random_state=seed)
    raise ValueError(f"невідома модель схильності: {model}")


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
        oob = np.where(np.isnan(oob), T.mean(), oob)  # об'єкт, що потрапив у всі бутстреп-вибірки
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
