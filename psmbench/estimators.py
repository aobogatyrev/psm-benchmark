"""Єдиний інтерфейс естиматорів ATT.

Конфігурація — словник:
  id, family, estimator ∈ {naive, psm_nn, psm_kernel, mahalanobis, ipw, aipw, eb},
  ps_model ∈ {lr, lr_dw, gbm, rf, None}, params {...}, se_method ∈ {analytic, ai, bootstrap}.

Усі естиматори повертають вектор ваг w над усіма спостереженнями; ATT = w-середнє Y
серед treated − w-середнє Y серед control. Для AIPW точкова оцінка містить поправку
на модель результату, але ваги для балансу — ті самі IPW-ваги.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .balance import balance_summary, effective_sample_size, smd_table
from .inference import abadie_imbens_se, stratified_bootstrap
from .matching import kernel_match, mahalanobis_match, nn_match
from .propensity import fit_propensity
from .weighting import entropy_balancing, ipw_att_weights
from .extended import causal_forest_att, dml_irm_att, xlearner_att


@dataclass
class EstimateResult:
    att: float
    se: float
    se_method: str
    ci_low: float
    ci_high: float
    naive: float
    n_treated: int
    n_control: int
    n_treated_used: int
    ess_control: float
    ps_metrics: Dict[str, float]
    balance: Dict[str, float]
    smd: pd.DataFrame
    weights: Optional[np.ndarray]
    info: Dict[str, object] = field(default_factory=dict)
    runtime_s: float = 0.0
    cate: Optional[np.ndarray] = None


def _weighted_att(Y: np.ndarray, T: np.ndarray, w: np.ndarray) -> float:
    tr, co = (T == 1) & (w > 0), (T == 0) & (w > 0)
    return float(np.average(Y[tr], weights=w[tr]) - np.average(Y[co], weights=w[co]))


def _outcome_model(kind: str, seed: int):
    if kind == "linear":
        return LinearRegression()
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=500,
                                             early_stopping=True, n_iter_no_change=30, random_state=seed)
    raise ValueError(kind)


def _point(df: pd.DataFrame, spec, cfg: dict, seed: int, want_details: bool = False):
    """Точкова оцінка. Повертає (att, w, details)."""
    p = cfg.get("params", {})
    est = cfg["estimator"]
    T = df[spec.treatment].astype(int).values
    Y = df[spec.outcome].astype(float).values
    X = df[spec.covariates].astype(float).values
    details: Dict[str, object] = {}
    ps = None
    if cfg.get("ps_model"):
        ps = fit_propensity(df, spec, cfg["ps_model"], seed=seed)
        details["ps_metrics"] = ps.metrics
        details["score"] = ps.logit

    if est == "naive":
        w = np.ones(len(T))
        used = T == 1

    elif est == "psm_nn":
        cal = p.get("caliper_sd")
        caliper = None if cal is None else cal * ps.logit.std(ddof=1)
        mr = nn_match(ps.logit, T, k=p.get("k", 1), caliper=caliper,
                      replacement=p.get("replacement", True), seed=seed)
        w = mr.control_weights(len(T)); w[mr.used_treated] = 1.0
        used = mr.used_treated
        details["match"] = mr

    elif est == "psm_kernel":
        mr = kernel_match(ps.e, T, bandwidth=p.get("bandwidth", 0.06), kernel=p.get("kernel", "epanechnikov"),
                          common_support=p.get("common_support", True))
        w = mr.control_weights(len(T)); w[mr.used_treated] = 1.0
        used = mr.used_treated
        details["match"] = mr

    elif est == "mahalanobis":
        Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        mr = mahalanobis_match(Xs, T, k=p.get("k", 1), replacement=p.get("replacement", True), seed=seed)
        w = mr.control_weights(len(T)); w[mr.used_treated] = 1.0
        used = mr.used_treated
        details["match"] = mr
        details["score"] = None

    elif est in ("ipw", "aipw"):
        w = ipw_att_weights(ps.e, T, trim=p.get("trim", 0.01))
        used = (T == 1) & (w > 0)

    elif est == "eb":
        cont = np.array([c in spec.continuous for c in spec.covariates])
        w, info = entropy_balancing(X, T, cont)
        used = T == 1
        details["eb"] = info

    elif est == "dml_irm":
        att, se, w, _ = dml_irm_att(X, Y, T, seed=seed, n_folds=p.get("crossfit", 5), trimming=p.get("trim", 0.01))
        details["se_analytic"] = se
        used = T == 1

    elif est == "xlearner":
        att, _, w, cate = xlearner_att(X, Y, T, ps.e, seed=seed)
        details["cate"] = cate
        used = T == 1

    elif est == "causal_forest":
        att, se, w, cate = causal_forest_att(X, Y, T, seed=seed, n_estimators=p.get("n_estimators", 1000))
        details["se_analytic"] = se
        details["cate"] = cate
        used = T == 1

    else:
        raise ValueError(f"невідомий естиматор {est}")

    if est == "aipw":
        att = _aipw_att(X, Y, T, ps.e, w, p, seed)
    elif est not in ("dml_irm", "xlearner", "causal_forest"):
        att = _weighted_att(Y, T, w)

    details["used_treated"] = used
    return att, w, details


def _aipw_att(X, Y, T, e, w, p, seed):
    """Подвійно робастна оцінка ATT (нормовані ваги):
       τ = mean_{T=1}(Y − m0(X)) − Σ_{T=0} w_j (Y_j − m0(X_j)) / Σ_{T=0} w_j,  w_j = e/(1−e).
    m0 — модель результату на контролі; crossfit=K>1 → K-fold перехресне підганяння (для ML)."""
    kind = p.get("outcome_model", "linear")
    K = p.get("crossfit", 1)
    co = T == 0
    m0 = np.empty(len(Y))
    if K <= 1:
        mdl = _outcome_model(kind, seed).fit(X[co], Y[co])
        m0[:] = mdl.predict(X)
    else:
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X):
            fit_idx = tr_idx[co[tr_idx]]
            mdl = _outcome_model(kind, seed).fit(X[fit_idx], Y[fit_idx])
            m0[te_idx] = mdl.predict(X[te_idx])
    tr = (T == 1) & (w > 0)
    cw = co & (w > 0)
    return float((Y[tr] - m0[tr]).mean() - np.average(Y[cw] - m0[cw], weights=w[cw]))


def estimate(df: pd.DataFrame, spec, cfg: dict, seed: int = 0, bootstrap_B: int = 500) -> EstimateResult:
    t0 = time.time()
    T = df[spec.treatment].astype(int).values
    Y = df[spec.outcome].astype(float).values
    att, w, d = _point(df, spec, cfg, seed, want_details=True)
    used = d["used_treated"]
    naive = float(Y[T == 1].mean() - Y[T == 0].mean())

    se_method = cfg.get("se_method", "bootstrap")
    info: Dict[str, object] = {}
    bootstrap_B = min(cfg.get("bootstrap_B", bootstrap_B), bootstrap_B)  # ліміт конфігурації — верхня межа
    if se_method == "analytic" and cfg["estimator"] == "naive":
        v = Y[T == 1].var(ddof=1) / (T == 1).sum() + Y[T == 0].var(ddof=1) / (T == 0).sum()
        se = float(np.sqrt(v)); ci = (att - 1.96 * se, att + 1.96 * se)
    elif se_method == "analytic" and "se_analytic" in d:
        se = float(d["se_analytic"]); ci = (att - 1.96 * se, att + 1.96 * se)
    elif se_method == "ai":
        score = d.get("score")
        if score is None:  # Mahalanobis: сусіди для σ² — за першою головною компонентою стандартизованих X
            X = df[spec.covariates].astype(float).values
            Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
            score = Xs @ np.linalg.svd(Xs, full_matrices=False)[2][0]
        se = abadie_imbens_se(Y, T, score, w, used, J=1)
        ci = (att - 1.96 * se, att + 1.96 * se)
    elif se_method == "bootstrap":
        se, ci_pct, ests = stratified_bootstrap(
            df, spec.treatment, lambda b, s: _point(b, spec, cfg, s)[0], B=bootstrap_B, seed=seed)
        ci = (att - 1.96 * se, att + 1.96 * se)
        info["ci_pct_low"], info["ci_pct_high"] = ci_pct
        info["bootstrap_valid"] = int(np.isfinite(ests).sum())
    else:
        se, ci = np.nan, (np.nan, np.nan)

    if w is None:  # метод без вагового подання (causal forest): баланс не визначено
        tab = smd_table(df, T, np.ones(len(T)), spec.covariates, spec.continuous)
        tab["smd_after"] = np.nan
    else:
        tab = smd_table(df, T, w, spec.covariates, spec.continuous)
    if "eb" in d:
        info.update({f"eb_{k}": v for k, v in d["eb"].items()})
    return EstimateResult(
        att=att, se=se, se_method=se_method, ci_low=ci[0], ci_high=ci[1], naive=naive,
        n_treated=int((T == 1).sum()), n_control=int((T == 0).sum()),
        n_treated_used=int(used.sum()), ess_control=effective_sample_size(w[T == 0]) if w is not None else np.nan,
        ps_metrics=d.get("ps_metrics", {}), balance=balance_summary(tab), smd=tab, weights=w,
        info=info, runtime_s=time.time() - t0, cate=d.get("cate"),
    )
