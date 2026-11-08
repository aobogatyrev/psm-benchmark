"""Extended estimators (block X of the protocol): DML-IRM, X-learner, causal forest.

All three return (att, se_or_None, w_or_None, cate_or_None):
  att  — ATT estimate;
  se   — analytic SE if the method provides one (DML, causal forest); otherwise None → bootstrap in estimators;
  w    — weight vector for the balance diagnostics (IPW weights from the method's own e(X)), or None;
  cate — individual effects τ̂(X_i) for PEHE (X-learner and causal forest only).

X1  DML-IRM, score='ATTE' (Chernozhukov et al. 2018; implementation of the doubleml package):
    nuisances g(X,T)=E[Y|X,T] and m(X)=P(T=1|X) — GBM, 5-fold cross-fitting, influence-function SE.
X3  X-learner (Künzel et al. 2019), GBM base learners; τ(x) = e(x)·τ0(x) + (1−e(x))·τ1(x);
    ATT = mean τ̂(X_i) over the treated. SE — bootstrap (expensive: 5 GBMs per resample).
X4  Causal forest (Athey, Tibshirani, Wager 2019) as CausalForestDML of the econml package:
    orthogonalisation on GBM nuisances, 1,000 honest trees; ATT and SE — effect_inference on the treated.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _gbm_reg(seed: int):
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=500,
                                         early_stopping=True, n_iter_no_change=30, random_state=seed)


def _gbm_clf(seed: int):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=500,
                                          early_stopping=True, n_iter_no_change=30, random_state=seed)


def dml_irm_att(X: np.ndarray, Y: np.ndarray, T: np.ndarray, seed: int = 0, n_folds: int = 5,
                trimming: float = 0.01) -> Tuple[float, float, Optional[np.ndarray], None]:
    import doubleml as dml
    import pandas as pd

    cols = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = Y
    df["d"] = T.astype(int)
    data = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=cols)
    np.random.seed(seed)
    m = dml.DoubleMLIRM(data, ml_g=_gbm_reg(seed), ml_m=_gbm_clf(seed), n_folds=n_folds,
                        score="ATTE", trimming_threshold=trimming)
    m.fit()
    att = float(np.asarray(m.coef).ravel()[0])
    se = float(np.asarray(m.se).ravel()[0])
    # weights for balance — IPW from the cross-fitted m(X)
    e = None
    try:
        e = np.asarray(m.predictions["ml_m"]).reshape(len(T), -1).mean(axis=1)
    except Exception:
        pass
    w = None
    if e is not None:
        e = np.clip(e, trimming, 1 - trimming)
        w = np.where(T == 1, 1.0, e / (1 - e))
    return att, se, w, None


def xlearner_att(X: np.ndarray, Y: np.ndarray, T: np.ndarray, e: np.ndarray, seed: int = 0
                 ) -> Tuple[float, None, np.ndarray, np.ndarray]:
    tr, co = T == 1, T == 0
    mu0 = _gbm_reg(seed).fit(X[co], Y[co])
    mu1 = _gbm_reg(seed + 1).fit(X[tr], Y[tr])
    d1 = Y[tr] - mu0.predict(X[tr])            # imputed effects for the treated
    d0 = mu1.predict(X[co]) - Y[co]            # for the controls
    tau1 = _gbm_reg(seed + 2).fit(X[tr], d1)
    tau0 = _gbm_reg(seed + 3).fit(X[co], d0)
    cate = e * tau0.predict(X) + (1 - e) * tau1.predict(X)
    att = float(cate[tr].mean())
    w = np.where(T == 1, 1.0, e / (1 - e))
    return att, None, w, cate


def causal_forest_att(X: np.ndarray, Y: np.ndarray, T: np.ndarray, seed: int = 0, n_estimators: int = 1000
                      ) -> Tuple[float, float, None, np.ndarray]:
    from econml.dml import CausalForestDML

    cf = CausalForestDML(model_y=_gbm_reg(seed), model_t=_gbm_clf(seed), discrete_treatment=True,
                         cv=5, n_estimators=n_estimators, min_samples_leaf=5, max_samples=0.45,
                         honest=True, random_state=seed, n_jobs=1)
    cf.fit(Y, T.astype(int), X=X)
    cate = np.asarray(cf.effect(X)).ravel()
    inf = cf.effect_inference(X[T == 1]).population_summary()
    att = float(np.asarray(inf.mean_point).ravel()[0])
    se = float(np.asarray(inf.stderr_mean).ravel()[0])
    return att, se, None, cate
