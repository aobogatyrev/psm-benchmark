"""make_figures.py — рисунки F1–F5 статті з results/ (PDF + PNG 300 dpi).

    python make_figures.py [--results results] [--out results/figures]

F1 перекриття схильності (LR) — LaLonde-DW та IHDP rep 1
F2 Love plot — LaLonde-DW: |SMD| до та після M1, M9, M6, M8
F3 forest plot ATT з 95 % CI — LaLonde-DW, усі конфігурації ядра, лінія 1 794 $
F4 розподіл зміщення по реплікаціях IHDP — боксплоти (якщо є результати ihdp)
F5 чутливість PSM 1:1 до caliper — ATT ± CI та частка збережених treated (якщо є S-* конфігурації)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psmbench import CORE_CONFIGS, EXTENDED_CONFIGS, load_dataset  # noqa: E402
from psmbench.propensity import fit_propensity  # noqa: E402

FAMILY_COLOR = {"baseline": "#52514e", "matching": "#2a78d6", "weighting": "#eb6834",
                "doubly_robust": "#1baf7a", "balancing": "#1baf7a", "cate": "#6250d6"}
PS_MARKER = {"": "o", "lr": "o", "gbm": "^", "rf": "s", "lr_dw": "D"}
FAMILY_UA = {"baseline": "базова", "matching": "зіставлення", "weighting": "зважування",
             "doubly_robust": "подвійно робастні", "balancing": "балансування", "cate": "CATE-методи"}
LABEL = {
    "M0": "Наївна різниця", "M1": "PSM 1:1 (LR)", "M2": "PSM 1:1 без повт. (LR)", "M3": "PSM 3:1 (LR)",
    "M4": "PSM ядерне (LR)", "M5": "Mahalanobis 1:1", "M6": "IPW (LR)", "M7": "AIPW (LR)",
    "M8": "Entropy balancing", "M9": "PSM 1:1 (GBM)", "M10": "PSM ядерне (GBM)", "M11": "IPW (GBM)",
    "M12": "PSM 1:1 (RF)", "M13": "PSM ядерне (RF)", "M14": "IPW (RF)", "M1+": "PSM 1:1 (LR, специф. DW)",
    "X1": "DML-IRM (GBM)", "X2": "AIPW (GBM, cross-fit)", "X3": "X-learner (GBM)", "X4": "Causal forest",
}
ORDER = [c["id"] for c in CORE_CONFIGS] + ["X1", "X2", "X3", "X4"]

# --lang en: англійські підписи для рукопису (Вісник, EN-версія); за замовчуванням — українські
LANG = "uk"
EN = {
    "Наївна різниця": "Naive difference", "PSM 1:1 без повт. (LR)": "PSM 1:1 w/o replacement (LR)",
    "PSM ядерне (LR)": "PSM kernel (LR)", "PSM ядерне (GBM)": "PSM kernel (GBM)", "PSM ядерне (RF)": "PSM kernel (RF)",
    "PSM 1:1 (LR, специф. DW)": "PSM 1:1 (LR, DW spec.)", "DML-IRM (GBM)": "DML-IRM (GBM)",
    "базова": "baseline", "зіставлення": "matching", "зважування": "weighting",
    "подвійно робастні": "doubly robust", "балансування": "balancing", "CATE-методи": "CATE methods",
    "подвійно робастні / балансування": "doubly robust / balancing", "LR, специф. DW": "LR, DW spec.",
    "контроль": "control", "щільність": "density", "IHDP, реплікація 1": "IHDP, replication 1",
    "до зіставлення": "before adjustment", "|SMD| (логарифмічна шкала; поріг 0.1)": "|SMD| (log scale; threshold 0.1)",
    "експеримент: ": "experimental benchmark: ", "ATT, \\$ (середнє по seeds; 95 % CI). Наївна різниця ": "ATT, USD (mean over seeds; 95 % CI). Naive difference ",
    " — поза межами осі": " is off the axis", "ATT̂ − 4 (по реплікаціях)": "estimated ATT − 4 (across replications)",
    "IHDP: зміщення по реплікаціях; над боксами — покриття 95 % CI": "IHDP: bias across replications; numbers above boxes — 95 % CI coverage",
    "з повт. вибором": "with replacement", "без повт. вибору": "without replacement", "ATT, $": "ATT, USD",
    "частка treated": "share of treated retained", "без": "none", "caliper, у SD логіту e(X)": "caliper, in SD of logit e(X)",
    "поза support": "outside support",
    "(а) caliper і повторний вибір, PSM 1:1": "(a) caliper and replacement, PSM 1:1",
    "кількість сусідів k": "number of matches k", "(б) кількість сусідів, PSM k:1 з повт. вибором": "(b) number of matches, PSM k:1 with replacement",
    "ширина вікна h": "bandwidth h", "(в) ширина вікна, ядерне зіставлення": "(c) bandwidth, kernel matching",
    "поріг обрізання (вага 0 при e(X) > 1 − trim)": "trimming threshold (weight 0 if e(X) > 1 − trim)", "(г) обрізання, IPW": "(d) trimming, IPW",
}


def t(s: str) -> str:
    return EN.get(s, s) if LANG == "en" else s

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9a9a95", "figure.dpi": 110})


def savefig(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def f1_overlap(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, (name, rep, title) in zip(axes, [("lalonde_dw_psid", None, "LaLonde-DW (PSID-1)"), ("ihdp", 1, t("IHDP, реплікація 1"))]):
        df, spec = load_dataset(name, rep=rep)
        ps = fit_propensity(df, spec, "lr", seed=0)
        T = df[spec.treatment].values
        bins = np.linspace(ps.logit.min(), ps.logit.max(), 40)
        ax.hist(ps.logit[T == 0], bins=bins, density=True, color="#52514e", alpha=0.45, label=t("контроль"))
        ax.hist(ps.logit[T == 1], bins=bins, density=True, color="#2a78d6", alpha=0.6, label="treated")
        lo, hi = ps.logit[T == 0].min(), ps.logit[T == 0].max()
        ax.axvspan(hi, ps.logit.max(), color="#e34948", alpha=0.08)
        ax.set_title(title, fontsize=9); ax.set_xlabel("logit e(X)"); ax.set_ylabel(t("щільність"))
        out_sup = float(((T == 1) & ((ps.logit < lo) | (ps.logit > hi))).mean())
        ax.text(0.98, 0.95, f"{t('поза support')}: {out_sup:.0%} treated\nAUC = {ps.metrics['auc']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[1].legend(frameon=False, loc="upper left")
    savefig(fig, out, "F1_overlap")


def f2_love(bal: pd.DataFrame, out: Path, dataset="lalonde_dw_psid"):
    b = bal[bal.dataset == dataset]
    if b.empty:
        return
    cfgs = [("M1", "PSM 1:1 (LR)"), ("M9", "PSM 1:1 (GBM)"), ("M6", "IPW (LR)"), ("M8", "Entropy balancing")]  # мовно нейтральні
    cov = ["re74", "re75", "age", "educ", "black", "married", "nodegree", "hispan"]
    y = np.arange(len(cov)); FLOOR = 1e-3
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    before = b[b.config_id == "M1"].groupby("covariate")["smd_before"].mean().abs().reindex(cov).clip(lower=FLOOR)
    ax.scatter(before.values, y, marker="o", s=34, facecolors="none", edgecolors="#52514e", label=t("до зіставлення"), zorder=3)
    fam = {c["id"]: c["family"] for c in CORE_CONFIGS}
    psm = {c["id"]: c.get("ps_model") or "" for c in CORE_CONFIGS}
    for cid, lab in cfgs:
        a = b[b.config_id == cid].groupby("covariate")["smd_after"].mean().abs().reindex(cov).clip(lower=FLOOR)
        if a.isna().all():
            continue
        ax.scatter(a.values, y, marker=PS_MARKER[psm[cid]], s=30, color=FAMILY_COLOR[fam[cid]], label=lab, zorder=4,
                   alpha=0.9, edgecolors="white", linewidths=0.6)
    ax.axvline(0.1, color="#9a9a95", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(cov); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(FLOOR * 0.7, 3)
    ax.set_xticks([FLOOR, 0.01, 0.1, 1]); ax.set_xticklabels([f"≤{FLOOR:g}", "0.01", "0.1", "1"])
    ax.set_xlabel(t("|SMD| (логарифмічна шкала; поріг 0.1)"))
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=1.0)
    savefig(fig, out, f"F2_love_{dataset}")


def f3_forest(runs: pd.DataFrame, out: Path, dataset="lalonde_dw_psid", benchmark=1794.34):
    r = runs[(runs.dataset == dataset) & (runs.config_id.isin(ORDER)) & (runs.config_id != "M0")]
    if r.empty:
        return
    g = r.groupby("config_id").agg(att=("att", "mean"), lo=("ci_low", "mean"), hi=("ci_high", "mean"),
                                    family=("family", "first"), ps=("ps_model", "first")).reindex([c for c in ORDER if c != "M0"]).dropna(subset=["att"])
    g["ps"] = g["ps"].fillna("")
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(g) + 1.4))
    y = np.arange(len(g))[::-1]
    for yi, (cid, row) in zip(y, g.iterrows()):
        col = FAMILY_COLOR[row.family]
        ax.plot([row.lo, row.hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
        ax.scatter(row.att, yi, marker=PS_MARKER[row.ps], s=38, color=col, zorder=4, edgecolors="white", linewidths=0.6)
    ax.axvline(benchmark, color="#0b0b0b", ls="--", lw=1)
    ax.text(benchmark, len(g) - 0.4, f"{t('експеримент: ')}{benchmark:,.0f} $".replace(",", "\u202f"), ha="center", va="bottom", fontsize=8)
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([t(LABEL.get(c, c)) for c in g.index])
    naive = runs[(runs.dataset == dataset) & (runs.config_id == "M0")]["att"].mean()
    naive_s = f"{naive:,.0f}".replace(",", "\u202f")
    ax.set_xlabel(f"{t('ATT, \\$ (середнє по seeds; 95 % CI). Наївна різниця ')}{naive_s} \\${t(' — поза межами осі')}")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=2, label=t(FAMILY_UA[f])) for f in ("matching", "weighting", "doubly_robust", "cate")]
    handles[2].set_label(t("подвійно робастні / балансування"))
    handles += [Line2D([], [], marker=m, color="#52514e", ls="", label=l) for m, l in (("o", "LR"), ("^", "GBM"), ("s", "RF"), ("D", t("LR, специф. DW")))]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=4,
              columnspacing=1.2, handletextpad=0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "\u202f")))
    savefig(fig, out, f"F3_forest_{dataset}")


def f4_ihdp_bias(runs: pd.DataFrame, out: Path):
    r = runs[(runs.dataset == "ihdp") & (runs.config_id.isin(ORDER))]
    if r.empty:
        return
    ids = [c for c in ORDER if c in set(r.config_id)]
    data = [r[r.config_id == c]["bias"].dropna().values for c in ids]
    fam = {c["id"]: c["family"] for c in CORE_CONFIGS + EXTENDED_CONFIGS}
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    bp = ax.boxplot(data, positions=range(len(ids)), widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": "#0b0b0b", "lw": 1.2})
    for patch, c in zip(bp["boxes"], ids):
        patch.set_facecolor(FAMILY_COLOR[fam[c]]); patch.set_alpha(0.55); patch.set_edgecolor("none")
    ax.axhline(0, color="#0b0b0b", ls="--", lw=1)
    cov = r.groupby("config_id")["covered"].mean().reindex(ids)
    for i, c in enumerate(ids):
        if np.isfinite(cov[c]):
            ax.text(i, ax.get_ylim()[1], f"{cov[c]:.0%}", ha="center", va="bottom", fontsize=7, color="#52514e")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels([t(LABEL.get(c, c)) for c in ids], rotation=45, ha="right")
    ax.set_ylabel(t("ATT̂ − 4 (по реплікаціях)"))
    ax.set_title(t("IHDP: зміщення по реплікаціях; над боксами — покриття 95 % CI"), fontsize=9, loc="left", pad=16)
    savefig(fig, out, "F4_ihdp_bias")


def f5_sensitivity(runs: pd.DataFrame, out: Path, dataset="lalonde_dw_psid", benchmark=1794.34):
    r = runs[(runs.dataset == dataset) & runs.config_id.str.startswith("S-M1-cal")].copy()
    base = runs[(runs.dataset == dataset) & (runs.config_id == "M1")]
    if r.empty:
        return
    r["params_d"] = r["params"].map(json.loads)
    r["cal"] = r.params_d.map(lambda p: p.get("caliper_sd")); r["repl"] = r.params_d.map(lambda p: p.get("replacement", True))
    if not base.empty:
        b = base.copy(); b["cal"] = 0.2; b["repl"] = True; r = pd.concat([r, b], ignore_index=True)
    POS = {0.1: 0, 0.2: 1, 0.5: 2, 1.0: 3, None: 4}  # рівновіддалені позиції; «без caliper» праворуч
    r["cal_x"] = r["cal"].map(lambda c: POS[None] if c is None or (isinstance(c, float) and np.isnan(c)) else POS[c])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    a1, ak, ab, atr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    # (a) caliper × повторний вибір; під точками «без повт.» — частка treated, що знайшли пару
    for repl, col, lab in ((True, "#2a78d6", t("з повт. вибором")), (False, "#eb6834", t("без повт. вибору"))):
        g = r[r.repl == repl].groupby("cal_x").agg(att=("att", "mean"), lo=("ci_low", "mean"), hi=("ci_high", "mean"),
                                                    frac=("frac_treated_used", "mean")).sort_index()
        if g.empty:
            continue
        a1.plot(g.index, g.att, "-o", color=col, lw=2, ms=5, label=lab)
        a1.fill_between(g.index, g.lo, g.hi, color=col, alpha=0.12)
        if not repl:
            for x, (att, fr) in zip(g.index, zip(g.att, g.frac)):
                a1.annotate(f"{fr:.0%}", (x, att), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7, color=col)
    a1.axhline(benchmark, color="#0b0b0b", ls="--", lw=1); a1.set_ylabel(t("ATT, $")); a1.legend(frameon=False, fontsize=8, loc="lower left")
    a1.set_xticks([0, 1, 2, 3, 4]); a1.set_xticklabels(["0.1", "0.2", "0.5", "1.0", t("без")])
    a1.set_xlabel(t("caliper, у SD логіту e(X)")); a1.set_title(t("(а) caliper і повторний вибір, PSM 1:1"), fontsize=9, loc="left")

    def _series(ax, ids: list, key: str, default, xlabel: str, title: str, fmt=lambda v: f"{v:g}"):
        """Панель ATT проти одного параметра: конфігурації ids (ядро + S-*), значення параметра з params."""
        s = runs[(runs.dataset == dataset) & runs.config_id.isin(ids)].copy()
        if s.empty:
            return
        s["params_d"] = s["params"].map(json.loads)
        s["v"] = s.params_d.map(lambda p: p.get(key, default))
        g = s.groupby("v").agg(att=("att", "mean"), lo=("ci_low", "mean"), hi=("ci_high", "mean")).sort_index()
        xs = np.arange(len(g))
        ax.plot(xs, g.att, "-o", color="#2a78d6", lw=2, ms=5)
        ax.fill_between(xs, g.lo, g.hi, color="#2a78d6", alpha=0.12)
        ax.axhline(benchmark, color="#0b0b0b", ls="--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels([fmt(v) for v in g.index]); ax.set_xlabel(xlabel); ax.set_ylabel(t("ATT, $"))
        ax.set_title(title, fontsize=9, loc="left")

    _series(ak, ["M1", "M3", "S-M1-k5", "S-M1-k10"], "k", 1, t("кількість сусідів k"),
            t("(б) кількість сусідів, PSM k:1 з повт. вибором"), fmt=lambda v: f"{int(v)}")
    _series(ab, ["M4", "S-M4-bw0.02", "S-M4-bw0.1", "S-M4-bw0.2"], "bandwidth", 0.06, t("ширина вікна h"),
            t("(в) ширина вікна, ядерне зіставлення"))
    _series(atr, ["M6", "S-M6-trim0.0", "S-M6-trim0.05"], "trim", 0.01, t("поріг обрізання (вага 0 при e(X) > 1 − trim)"),
            t("(г) обрізання, IPW"))
    fig.tight_layout()
    savefig(fig, out, f"F5_sensitivity_{dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lang", default="uk", choices=["uk", "en"], help="мова підписів")
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    res = Path(a.results)
    out = Path(a.out or res / ("figures_en" if a.lang == "en" else "figures")); out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(res / "runs.csv") if (res / "runs.csv").exists() else pd.DataFrame()
    bal = pd.read_csv(res / "balance.csv") if (res / "balance.csv").exists() else pd.DataFrame()
    f1_overlap(out)
    if not bal.empty:
        f2_love(bal, out)
    if not runs.empty:
        runs["ps_model"] = runs["ps_model"].fillna("")
        f3_forest(runs, out); f4_ihdp_bias(runs, out); f5_sensitivity(runs, out)


if __name__ == "__main__":
    main()
