"""Quick correctness checks (pytest -q tests/). Each takes < 30 s."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psmbench import estimate, load_dataset, CORE_CONFIGS  # noqa: E402
from psmbench.weighting import entropy_balancing  # noqa: E402

CFG = {c["id"]: c for c in CORE_CONFIGS}


def test_nsw_experiment_naive_is_benchmark():
    df, spec = load_dataset("lalonde_nsw_exp")
    r = estimate(df, spec, CFG["M0"], seed=1, bootstrap_B=10)
    assert abs(r.att - 1794.34) < 0.1
    assert r.n_treated == 185 and r.n_control == 260


def test_ipw_and_eb_recover_experiment_within_se():
    df, spec = load_dataset("lalonde_nsw_exp")
    for cid in ("M6", "M8"):
        r = estimate(df, spec, CFG[cid], seed=1, bootstrap_B=20)
        assert abs(r.att - 1794.34) < 2 * 700, cid   # experiment: SE ≈ 670


def test_entropy_balancing_exact_first_moments():
    df, spec = load_dataset("lalonde_dw_psid")
    X = df[spec.covariates].values.astype(float)
    T = df[spec.treatment].values
    cont = np.array([c in spec.continuous for c in spec.covariates])
    w, info = entropy_balancing(X, T, cont)
    wt = (X[T == 0] * w[T == 0][:, None]).sum(0)
    assert np.allclose(wt, X[T == 1].mean(0), rtol=1e-4, atol=1e-4)
    assert info["converged"]


def test_psm_weights_sum_and_balance_improves():
    df, spec = load_dataset("lalonde_dw_psid")
    r = estimate(df, spec, CFG["M1"], seed=1, bootstrap_B=10)
    T = df[spec.treatment].values
    assert np.isclose(r.weights[T == 0].sum(), r.n_treated_used)  # every treated unit → one control with total weight 1
    assert r.balance["mean_abs_smd_after"] < r.balance["mean_abs_smd_before"] / 3
    assert np.isfinite(r.se) and r.se > 0


def test_ihdp_true_att_is_four():
    df, spec = load_dataset("ihdp", rep=7)
    assert abs(spec.true_att - 4.0) < 1e-6
    assert len(df) == 747 and (df.treat == 1).sum() == 139
    r = estimate(df, spec, CFG["M4"], seed=1, bootstrap_B=10)
    assert abs(r.att - 4.0) < 1.0


def test_bootstrap_runs_for_kernel():
    df, spec = load_dataset("ihdp", rep=3)
    r = estimate(df, spec, CFG["M4"], seed=1, bootstrap_B=15)
    assert r.info["bootstrap_valid"] >= 10 and np.isfinite(r.se)


def test_extended_estimators_on_ihdp():
    from psmbench import EXTENDED_CONFIGS
    cfg = {c["id"]: c for c in EXTENDED_CONFIGS}
    df, spec = load_dataset("ihdp", rep=5)
    truth = (df.mu1 - df.mu0).values
    for cid in ("X1", "X4"):          # analytic SE, no bootstrap
        r = estimate(df, spec, cfg[cid], seed=1, bootstrap_B=5)
        assert abs(r.att - 4.0) < 1.0 and np.isfinite(r.se) and r.se > 0, cid
    r = estimate(df, spec, cfg["X3"], seed=1, bootstrap_B=5)
    assert abs(r.att - 4.0) < 1.0 and r.cate is not None
    assert np.sqrt(np.mean((r.cate - truth) ** 2)) < 2.0   # PEHE
