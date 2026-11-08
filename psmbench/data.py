"""Завантаження наборів даних у єдиному вигляді: DataFrame + DatasetSpec."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LALONDE_COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
# Специфікація Dehejia & Wahba (1999) для моделі схильності (LR+):
LALONDE_DW_EXTRA = ["u74", "u75"]  # індикатори нульового доходу; квадрати додаються у propensity.py
IHDP_COVARIATES = [f"x{i}" for i in range(1, 26)]


@dataclass
class DatasetSpec:
    name: str
    treatment: str
    outcome: str
    covariates: List[str]
    continuous: List[str]              # для другого моменту в entropy balancing та SMD-знаменника
    true_att: Optional[float] = None   # відомий еталон (IHDP: з mu0/mu1; NSW-exp: 1794.34)
    rep: Optional[int] = None
    extra_ps_covariates: List[str] = field(default_factory=list)  # для LR+ (DW-специфікація)
    squared_terms: List[str] = field(default_factory=list)        # для LR+


_LALONDE_FILES = {
    "lalonde_dw_psid": "lalonde_dw_psid.csv",
    "lalonde_dw_cps": "lalonde_dw_cps.csv",
    "lalonde_nsw_exp": "lalonde_nsw_exp.csv",
}


def list_datasets() -> List[str]:
    return list(_LALONDE_FILES) + ["ihdp"]


def load_dataset(name: str, rep: Optional[int] = None, data_dir: Path = DATA_DIR):
    """Повертає (df, spec). Для ihdp потрібен номер реплікації 1..100."""
    if name in _LALONDE_FILES:
        df = pd.read_csv(data_dir / "lalonde" / _LALONDE_FILES[name])
        spec = DatasetSpec(
            name=name, treatment="treat", outcome="re78",
            covariates=LALONDE_COVARIATES,
            continuous=["age", "educ", "re74", "re75"],
            true_att=1794.34,   # експериментальний еталон DW 1999 (NSW treated − NSW control)
            extra_ps_covariates=LALONDE_DW_EXTRA,
            squared_terms=["age", "educ", "re74", "re75"],
        )
        return df.reset_index(drop=True), spec

    if name == "ihdp":
        if rep is None:
            raise ValueError("для ihdp вкажіть rep=1..100")
        all_ = _load_ihdp_all(data_dir)
        df = all_[all_["rep"] == rep].sort_values("idx").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"реплікація {rep} відсутня")
        true_att = float((df["mu1"] - df["mu0"])[df["treat"] == 1].mean())
        spec = DatasetSpec(
            name="ihdp", treatment="treat", outcome="y_factual",
            covariates=IHDP_COVARIATES,
            continuous=[f"x{i}" for i in range(1, 7)],
            true_att=true_att, rep=rep,
        )
        return df, spec

    raise ValueError(f"невідомий набір: {name}; доступні: {list_datasets()}")


_IHDP_CACHE = {}


def _load_ihdp_all(data_dir: Path) -> pd.DataFrame:
    key = str(data_dir)
    if key not in _IHDP_CACHE:
        _IHDP_CACHE[key] = pd.read_csv(data_dir / "ihdp" / "ihdp_npci_1-100.csv.gz")
    return _IHDP_CACHE[key]
