"""psmbench — a comparison of ATT estimators on the LaLonde and IHDP benchmarks.

Every estimator is reduced to one representation: a weight vector w over all observations,
ATT = weighted mean of Y among the treated − weighted mean of Y among the controls.
Balance (SMD), the effective sample size and the bootstrap are therefore computed identically
for matching, weighting, doubly robust and balancing methods.
"""
from .data import DatasetSpec, load_dataset, list_datasets  # noqa: F401
from .estimators import estimate, EstimateResult  # noqa: F401
from .configs import CORE_CONFIGS, EXTENDED_CONFIGS, SENSITIVITY_CONFIGS, get_configs  # noqa: F401

__version__ = "0.1.0"
