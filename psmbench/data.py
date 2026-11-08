"""Loading of the datasets in a single form: DataFrame + DatasetSpec."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LALONDE_COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
# Dehejia & Wahba (1999) specification of the propensity model (LR+):
LALONDE_DW_EXTRA = ["u74", "u75"]  # zero-earnings indicators; squared terms are added in propensity.py
IHDP_COVARIATES = [f"x{i}" for i in range(1, 26)]


@dataclass
class DatasetSpec:
    name: str
    treatment: str
    outcome: str
    covariates: List[str]
    continuous: List[str]              # for the second moments in entropy balancing and the SMD denominator
    true_att: Optional[float] = None   # known reference value (IHDP: from mu0/mu1; NSW experiment: 1794.34)
    rep: Optional[int] = None
    extra_ps_covariates: List[str] = field(default_factory=list)  # for LR+ (DW specification)
    squared_terms: List[str] = field(default_factory=list)        # for LR+


_LALONDE_FILES = {
    "lalonde_dw_psid": "lalonde_dw_psid.csv",
    "lalonde_dw_cps": "lalonde_dw_cps.csv",
    "lalonde_nsw_exp": "lalonde_nsw_exp.csv",
}


def list_datasets() -> List[str]:
    return list(_LALONDE_FILES) + ["ihdp"]


def load_dataset(name: str, rep: Optional[int] = None, data_dir: Path = DATA_DIR):
    """Return (df, spec). For ihdp the replication number 1..100 is required."""
    if name in _LALONDE_FILES:
        df = pd.read_csv(data_dir / "lalonde" / _LALONDE_FILES[name])
        spec = DatasetSpec(
            name=name, treatment="treat", outcome="re78",
            covariates=LALONDE_COVARIATES,
            continuous=["age", "educ", "re74", "re75"],
            true_att=1794.34,   # experimental benchmark of DW 1999 (NSW treated − NSW control)
            extra_ps_covariates=LALONDE_DW_EXTRA,
            squared_terms=["age", "educ", "re74", "re75"],
        )
        return df.reset_index(drop=True), spec

    if name == "ihdp":
        if rep is None:
            raise ValueError("for ihdp pass rep=1..100")
        all_ = _load_ihdp_all(data_dir)
        df = all_[all_["rep"] == rep].sort_values("idx").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"replication {rep} is missing")
        true_att = float((df["mu1"] - df["mu0"])[df["treat"] == 1].mean())
        spec = DatasetSpec(
            name="ihdp", treatment="treat", outcome="y_factual",
            covariates=IHDP_COVARIATES,
            continuous=[f"x{i}" for i in range(1, 7)],
            true_att=true_att, rep=rep,
        )
        return df, spec

    raise ValueError(f"unknown dataset: {name}; available: {list_datasets()}")


_IHDP_CACHE = {}


def _load_ihdp_all(data_dir: Path) -> pd.DataFrame:
    key = str(data_dir)
    if key not in _IHDP_CACHE:
        _IHDP_CACHE[key] = pd.read_csv(data_dir / "ihdp" / "ihdp_npci_1-100.csv.gz")
    return _IHDP_CACHE[key]
