"""make_tables.py — таблиці Т1–Т5 статті з results/runs.csv та results/balance.csv.

    python make_tables.py [--results results] [--out results/tables]

Виводить CSV і Markdown для кожної таблиці. Числа агрегуються по seeds (LaLonde) або по
реплікаціях (IHDP): середнє, а для ATT додатково SD між seeds як міра Монте-Карло-варіації.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psmbench import CORE_CONFIGS, load_dataset  # noqa: E402

ORDER = [c["id"] for c in CORE_CONFIGS] + ["X1", "X2", "X3", "X4"]
LABEL = {
    "M0": "Наївна різниця", "M1": "PSM 1:1 NN, з повт. вибором", "M2": "PSM 1:1 NN, без повт. вибору",
    "M3": "PSM 3:1 NN", "M4": "PSM ядерне", "M5": "Mahalanobis 1:1", "M6": "IPW", "M7": "AIPW",
    "M8": "Entropy balancing", "M9": "PSM 1:1 NN", "M10": "PSM ядерне", "M11": "IPW",
    "M12": "PSM 1:1 NN", "M13": "PSM ядерне", "M14": "IPW", "M1+": "PSM 1:1 NN (специф. DW)",
    "X1": "DML (PLR)", "X2": "AIPW (ML, cross-fit)", "X3": "X-learner", "X4": "Causal forest",
}
PS_LABEL = {"": "—", "lr": "LR", "lr_dw": "LR+", "gbm": "GBM", "rf": "RF"}
LABEL.update({"X1": "DML-IRM (ATTE, GBM)", "X2": "AIPW (GBM, cross-fit)", "X3": "X-learner (GBM)", "X4": "Causal forest"})


def _order(df, col="config_id"):
    df = df.copy()
    df["_o"] = df[col].map({k: i for i, k in enumerate(ORDER)}).fillna(999)
    return df.sort_values("_o").drop(columns="_o")


def save(df: pd.DataFrame, out: Path, name: str, floatfmt="%.3f"):
    df.to_csv(out / f"{name}.csv", index=False, float_format=floatfmt)
    with open(out / f"{name}.md", "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=floatfmt.replace("%", "")))
    print(f"\n{name}:\n{df.to_string(index=False)}")


def t1_datasets(out: Path):
    rows = []
    for name in ("lalonde_dw_psid", "lalonde_dw_cps", "lalonde_nsw_exp"):
        df, spec = load_dataset(name)
        T = df[spec.treatment].values; Y = df[spec.outcome].values
        rows.append({"Набір": name, "N treated": int((T == 1).sum()), "N control": int((T == 0).sum()),
                     "Коваріат": len(spec.covariates), "Результат": spec.outcome,
                     "Еталон ATT": spec.true_att, "Наївна різниця": Y[T == 1].mean() - Y[T == 0].mean()})
    df, spec = load_dataset("ihdp", rep=1)
    T = df[spec.treatment].values; Y = df[spec.outcome].values
    rows.append({"Набір": "ihdp (100 реплікацій)", "N treated": int((T == 1).sum()), "N control": int((T == 0).sum()),
                 "Коваріат": 25, "Результат": "y_factual", "Еталон ATT": 4.0,
                 "Наївна різниця": Y[T == 1].mean() - Y[T == 0].mean()})
    save(pd.DataFrame(rows), out, "T1_datasets", "%.2f")


def t2_balance(bal: pd.DataFrame, out: Path, dataset="lalonde_dw_psid", configs=("M1", "M4", "M6", "M8")):
    b = bal[(bal.dataset == dataset) & (bal.config_id.isin(configs))]
    if b.empty:
        return
    before = b[b.config_id == configs[0]].groupby("covariate")["smd_before"].mean()
    after = b.groupby(["covariate", "config_id"])["smd_after"].mean().unstack("config_id")
    after = after[[c for c in configs if c in after.columns]]
    after.columns = [f"після {LABEL[c]}" for c in after.columns]
    tab = pd.concat([before.rename("SMD до"), after], axis=1).reset_index()
    cov_order = {c: i for i, c in enumerate(["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"])}
    tab = tab.sort_values("covariate", key=lambda s: s.map(cov_order))
    save(tab, out, f"T2_balance_{dataset}")


def t3_lalonde(runs: pd.DataFrame, out: Path, dataset="lalonde_dw_psid"):
    r = runs[runs.dataset == dataset]
    if r.empty:
        return
    g = r.groupby(["config_id", "ps_model"], sort=False)
    tab = g.agg(att=("att", "mean"), att_sd_seeds=("att", "std"), se=("se", "mean"),
                ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"), bias=("bias", "mean"),
                covered=("covered", "mean"), frac_treated=("frac_treated_used", "mean"),
                ess_control=("ess_control", "mean"), mean_abs_smd=("mean_abs_smd_after", "mean"),
                max_abs_smd=("max_abs_smd_after", "mean"), n_seeds=("seed", "nunique")).reset_index()
    tab.insert(1, "Естиматор", tab.config_id.map(LABEL))
    tab["ps_model"] = tab.ps_model.fillna("").map(PS_LABEL)
    save(_order(tab), out, f"T3_main_{dataset}", "%.2f")


def t4_ihdp(runs: pd.DataFrame, out: Path):
    r = runs[runs.dataset == "ihdp"]
    if r.empty:
        return
    g = r.groupby(["config_id", "ps_model"], sort=False)
    tab = g.agg(bias=("bias", "mean"), sd=("att", "std"),
                rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                coverage=("covered", "mean"), ci_len=("ci_high", lambda x: np.nan),
                frac_treated=("frac_treated_used", "mean"), ess_control=("ess_control", "mean"),
                mean_abs_smd=("mean_abs_smd_after", "mean"), pehe=("pehe", "mean"),
                n_reps=("rep", "nunique")).reset_index()
    tab["ci_len"] = g.apply(lambda d: (d.ci_high - d.ci_low).mean()).values
    tab.insert(1, "Естиматор", tab.config_id.map(LABEL))
    tab["ps_model"] = tab.ps_model.fillna("").map(PS_LABEL)
    save(_order(tab), out, "T4_main_ihdp")


def t5_ps_quality(runs: pd.DataFrame, out: Path):
    # лише конфігурації ядра (M6/M11/M14): рядки сітки чутливості S-M6-trim* мають інший trim
    r = runs[(runs.estimator == "ipw") & ~runs.config_id.str.startswith("S-")].copy()
    r = r.dropna(subset=["ps_auc"])
    if r.empty:
        return
    g = r.groupby(["dataset", "ps_model"]).agg(
        auc=("ps_auc", "mean"), logloss=("ps_logloss", "mean"), brier=("ps_brier", "mean"),
        frac_e_gt_095=("ps_frac_e_gt_095", "mean"), max_abs_smd_after_ipw=("max_abs_smd_after", "mean"),
        ipw_bias=("bias", "mean")).reset_index()
    g["ps_model"] = g.ps_model.map(PS_LABEL)
    save(g, out, "T5_ps_quality")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = Path(a.results); out = Path(a.out or res / "tables"); out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(res / "runs.csv")
    bal = pd.read_csv(res / "balance.csv") if (res / "balance.csv").exists() else pd.DataFrame()
    runs["ps_model"] = runs["ps_model"].fillna("")
    t1_datasets(out)
    for ds in ("lalonde_dw_psid", "lalonde_dw_cps"):
        if not bal.empty:
            t2_balance(bal, out, ds)
        t3_lalonde(runs, out, ds)
    t4_ihdp(runs, out)
    t5_ps_quality(runs, out)


if __name__ == "__main__":
    main()
