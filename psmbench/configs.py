"""Estimator configurations (core M0–M14 + M1+, extensions X*, sensitivity grid S*).

Identifiers follow the experimental protocol. Parameters:
  caliper_sd   — caliper in units of the SD of the propensity logit (0.2 — Austin 2011)
  replacement  — matching with replacement of controls
  k            — controls per treated unit
  bandwidth    — kernel-matching bandwidth on the probability scale
  trim         — IPW: controls with e > 1−trim get weight 0
  outcome_model, crossfit — AIPW
"""
from __future__ import annotations

import copy
from typing import List

_PSM11 = {"k": 1, "caliper_sd": 0.2, "replacement": True}

CORE_CONFIGS: List[dict] = [
    {"id": "M0", "family": "baseline", "estimator": "naive", "ps_model": None, "params": {}, "se_method": "analytic"},
    {"id": "M1", "family": "matching", "estimator": "psm_nn", "ps_model": "lr", "params": dict(_PSM11), "se_method": "ai"},
    {"id": "M2", "family": "matching", "estimator": "psm_nn", "ps_model": "lr",
     "params": {"k": 1, "caliper_sd": 0.2, "replacement": False}, "se_method": "bootstrap"},
    {"id": "M3", "family": "matching", "estimator": "psm_nn", "ps_model": "lr",
     "params": {"k": 3, "caliper_sd": 0.2, "replacement": True}, "se_method": "ai"},
    {"id": "M4", "family": "matching", "estimator": "psm_kernel", "ps_model": "lr",
     "params": {"bandwidth": 0.06, "kernel": "epanechnikov"}, "se_method": "bootstrap"},
    {"id": "M5", "family": "matching", "estimator": "mahalanobis", "ps_model": None,
     "params": {"k": 1, "replacement": True}, "se_method": "ai"},
    {"id": "M6", "family": "weighting", "estimator": "ipw", "ps_model": "lr", "params": {"trim": 0.01}, "se_method": "bootstrap"},
    {"id": "M7", "family": "doubly_robust", "estimator": "aipw", "ps_model": "lr",
     "params": {"trim": 0.01, "outcome_model": "linear", "crossfit": 1}, "se_method": "bootstrap"},
    {"id": "M8", "family": "balancing", "estimator": "eb", "ps_model": None, "params": {}, "se_method": "bootstrap"},
    {"id": "M9", "family": "matching", "estimator": "psm_nn", "ps_model": "gbm", "params": dict(_PSM11), "se_method": "ai"},
    {"id": "M10", "family": "matching", "estimator": "psm_kernel", "ps_model": "gbm",
     "params": {"bandwidth": 0.06, "kernel": "epanechnikov"}, "se_method": "bootstrap"},
    {"id": "M11", "family": "weighting", "estimator": "ipw", "ps_model": "gbm", "params": {"trim": 0.01}, "se_method": "bootstrap"},
    {"id": "M12", "family": "matching", "estimator": "psm_nn", "ps_model": "rf", "params": dict(_PSM11), "se_method": "ai"},
    {"id": "M13", "family": "matching", "estimator": "psm_kernel", "ps_model": "rf",
     "params": {"bandwidth": 0.06, "kernel": "epanechnikov"}, "se_method": "bootstrap"},
    {"id": "M14", "family": "weighting", "estimator": "ipw", "ps_model": "rf", "params": {"trim": 0.01}, "se_method": "bootstrap"},
    # LaLonde only: Dehejia–Wahba specification
    {"id": "M1+", "family": "matching", "estimator": "psm_nn", "ps_model": "lr_dw", "params": dict(_PSM11),
     "se_method": "ai", "datasets": ["lalonde_dw_psid", "lalonde_dw_cps", "lalonde_nsw_exp"]},
]

EXTENDED_CONFIGS: List[dict] = [
    # Require doubleml (X1) and econml (X4): pip install doubleml econml
    {"id": "X1", "family": "doubly_robust", "estimator": "dml_irm", "ps_model": None,
     "params": {"trim": 0.01, "crossfit": 5}, "se_method": "analytic"},
    {"id": "X2", "family": "doubly_robust", "estimator": "aipw", "ps_model": "gbm",
     "params": {"trim": 0.01, "outcome_model": "gbm", "crossfit": 5}, "se_method": "bootstrap"},
    {"id": "X3", "family": "cate", "estimator": "xlearner", "ps_model": "gbm", "params": {},
     "se_method": "bootstrap", "bootstrap_B": 100},  # upper bound on B: 5 GBMs per resample
    {"id": "X4", "family": "cate", "estimator": "causal_forest", "ps_model": None,
     "params": {"n_estimators": 1000}, "se_method": "analytic"},
]


def _sens(base_id: str, suffix: str, **overrides) -> dict:
    base = copy.deepcopy(next(c for c in CORE_CONFIGS if c["id"] == base_id))
    base["id"] = f"S-{base_id}-{suffix}"
    base["params"].update(overrides)
    base["se_method"] = "ai" if base["estimator"] == "psm_nn" and base["params"].get("replacement", True) else "bootstrap"
    return base


SENSITIVITY_CONFIGS: List[dict] = (
    [_sens("M1", f"cal{c}", caliper_sd=c) for c in (None, 0.1, 0.5, 1.0)]
    + [_sens("M1", f"cal{c}-norepl", caliper_sd=c, replacement=False) for c in (None, 0.1, 0.2, 0.5, 1.0)]
    + [_sens("M1", f"k{k}", k=k) for k in (5, 10)]
    + [_sens("M4", f"bw{b}", bandwidth=b) for b in (0.02, 0.10, 0.20)]
    + [_sens("M6", f"trim{t}", trim=t) for t in (0.0, 0.05)]
)


def get_configs(name: str) -> List[dict]:
    return {"core": CORE_CONFIGS, "extended": EXTENDED_CONFIGS, "sensitivity": SENSITIVITY_CONFIGS,
            "all": CORE_CONFIGS + EXTENDED_CONFIGS + SENSITIVITY_CONFIGS}[name]
