"""psmbench — порівняння методів оцінювання ATT на бенчмарках LaLonde та IHDP.

Усі естиматори зводяться до одного подання: вектор ваг w над усіма спостереженнями,
ATT = зважене середнє Y серед treated − зважене середнє Y серед control.
Завдяки цьому баланс (SMD), ефективний розмір вибірки та бутстреп рахуються однаково
для зіставлення, зважування, подвійно робастних і балансувальних методів.
"""
from .data import DatasetSpec, load_dataset, list_datasets  # noqa: F401
from .estimators import estimate, EstimateResult  # noqa: F401
from .configs import CORE_CONFIGS, EXTENDED_CONFIGS, SENSITIVITY_CONFIGS, get_configs  # noqa: F401

__version__ = "0.1.0"
