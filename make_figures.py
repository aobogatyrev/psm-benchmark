"""make_figures.py — figures F1–F5 of the article from results/ (PDF + PNG 300 dpi).

    python make_figures.py [--results results] [--out results/figures] [--lang en|uk]

F1 propensity overlap (LR) — LaLonde-DW and IHDP rep 1
F2 Love plot — LaLonde-DW: |SMD| before and after M1, M9, M6, M8
F3 forest plot of ATT with 95 % CI — LaLonde-DW, all core configurations, line at 1,794 USD
F4 distribution of the bias across IHDP replications — box plots (if ihdp results exist)
F5 sensitivity of PSM to its design — caliper × replacement, k, bandwidth, trimming (if S-* configurations exist)

Labels are English by default; --lang uk writes the Ukrainian version to results/figures_uk.
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
FAMILY_LABEL = {
    'baseline': 'baseline', 'matching': 'matching', 'weighting': 'weighting',
    'doubly_robust': 'doubly robust', 'balancing': 'balancing', 'cate': 'CATE methods',
}
LABEL = {
    'M0': 'Naive difference', 'M1': 'PSM 1:1 (LR)', 'M2': 'PSM 1:1 w/o replacement (LR)', 'M3': 'PSM 3:1 (LR)',
    'M4': 'PSM kernel (LR)', 'M5': 'Mahalanobis 1:1', 'M6': 'IPW (LR)', 'M7': 'AIPW (LR)',
    'M8': 'Entropy balancing', 'M9': 'PSM 1:1 (GBM)', 'M10': 'PSM kernel (GBM)', 'M11': 'IPW (GBM)',
    'M12': 'PSM 1:1 (RF)', 'M13': 'PSM kernel (RF)', 'M14': 'IPW (RF)', 'M1+': 'PSM 1:1 (LR, DW spec.)',
    'X1': 'DML-IRM (GBM)', 'X2': 'AIPW (GBM, cross-fit)', 'X3': 'X-learner (GBM)', 'X4': 'Causal forest',
}
ORDER = [c["id"] for c in CORE_CONFIGS] + ["X1", "X2", "X3", "X4"]

# --lang uk: Ukrainian labels (English → Ukrainian); the default is English
LANG = "en"
UK = {
    'Naive difference': 'Наївна різниця', 'PSM 1:1 w/o replacement (LR)': 'PSM 1:1 без повт. (LR)',
    'PSM kernel (LR)': 'PSM ядерне (LR)', 'PSM kernel (GBM)': 'PSM ядерне (GBM)',
    'PSM kernel (RF)': 'PSM ядерне (RF)', 'PSM 1:1 (LR, DW spec.)': 'PSM 1:1 (LR, специф. DW)',
    'baseline': 'базова', 'matching': 'зіставлення',
    'weighting': 'зважування', 'doubly robust': 'подвійно робастні',
    'balancing': 'балансування', 'CATE methods': 'CATE-методи',
    'doubly robust / balancing': 'подвійно робастні / балансування', 'LR, DW spec.': 'LR, специф. DW',
    'control': 'контроль', 'density': 'щільність',
    'IHDP, replication 1': 'IHDP, реплікація 1', 'before adjustment': 'до зіставлення',
    '|SMD| (log scale; threshold 0.1)': '|SMD| (логарифмічна шкала; поріг 0.1)', 'experimental benchmark: ': 'експеримент: ',
    'ATT, USD (mean over seeds; 95 % CI). Naive difference ': 'ATT, \\$ (середнє по seeds; 95 % CI). Наївна різниця ', ' is off the axis': ' — поза межами осі',
    'estimated ATT − 4 (across replications)': 'ATT̂ − 4 (по реплікаціях)', 'IHDP: bias across replications; numbers above boxes — 95 % CI coverage': 'IHDP: зміщення по реплікаціях; над боксами — покриття 95 % CI',
    'with replacement': 'з повт. вибором', 'without replacement': 'без повт. вибору',
    'ATT, USD': 'ATT, $', 'share of treated retained': 'частка treated',
    'none': 'без', 'caliper, in SD of logit e(X)': 'caliper, у SD логіту e(X)',
    'outside support': 'поза support', '(a) caliper and replacement, PSM 1:1': '(а) caliper і повторний вибір, PSM 1:1',
    'number of matches k': 'кількість сусідів k', '(b) number of matches, PSM k:1 with replacement': '(б) кількість сусідів, PSM k:1 з повт. вибором',
    'bandwidth h': 'ширина вікна h', '(c) bandwidth, kernel matching': '(в) ширина вікна, ядерне зіставлення',
    'trimming threshold (weight 0 if e(X) > 1 − trim)': 'поріг обрізання (вага 0 при e(X) > 1 − trim)', '(d) trimming, IPW': '(г) обрізання, IPW',
}


def t(s: str) -> str:
    return UK.get(s, s) if LANG == "uk" else s

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
    for ax, (name, rep, title) in zip(axes, [("lalonde_dw_psid", None, "LaLonde-DW (PSID-1)"), ("ihdp", 1, t("IHDP, replication 1"))]):
        df, spec = load_dataset(name, rep=rep)
        ps = fit_propensity(df, spec, "lr", seed=0)
        T = df[spec.treatment].values
        bins = np.linspace(ps.logit.min(), ps.logit.max(), 40)
        ax.hist(ps.logit[T == 0], bins=bins, density=True, color="#52514e", alpha=0.45, label=t("control"))
        ax.hist(ps.logit[T == 1], bins=bins, density=True, color="#2a78d6", alpha=0.6, label="treated")
        lo, hi = ps.logit[T == 0].min(), ps.logit[T == 0].max()
        ax.axvspan(hi, ps.logit.max(), color="#e34948", alpha=0.08)
        ax.set_title(title, fontsize=9); ax.set_xlabel("logit e(X)"); ax.set_ylabel(t("density"))
        out_sup = float(((T == 1) & ((ps.logit < lo) | (ps.logit > hi))).mean())
        ax.text(0.98, 0.95, f"{t('outside support')}: {out_sup:.0%} treated\nAUC = {ps.metrics['auc']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[1].legend(frameon=False, loc="upper left")
    savefig(fig, out, "F1_overlap")


def f2_love(bal: pd.DataFrame, out: Path, dataset="lalonde_dw_psid"):
    b = bal[bal.dataset == dataset]
    if b.empty:
        return
    cfgs = [("M1", "PSM 1:1 (LR)"), ("M9", "PSM 1:1 (GBM)"), ("M6", "IPW (LR)"), ("M8", "Entropy balancing")]  # language-neutral labels
    cov = ["re74", "re75", "age", "educ", "black", "married", "nodegree", "hispan"]
    y = np.arange(len(cov)); FLOOR = 1e-3
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    before = b[b.config_id == "M1"].groupby("covariate")["smd_before"].mean().abs().reindex(cov).clip(lower=FLOOR)
    ax.scatter(before.values, y, marker="o", s=34, facecolors="none", edgecolors="#52514e", label=t("before adjustment"), zorder=3)
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
    ax.set_xlabel(t("|SMD| (log scale; threshold 0.1)"))
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
    ax.text(benchmark, len(g) - 0.4, f"{t('experimental benchmark: ')}{benchmark:,.0f} $".replace(",", "\u202f"), ha="center", va="bottom", fontsize=8)
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([t(LABEL.get(c, c)) for c in g.index])
    naive = runs[(runs.dataset == dataset) & (runs.config_id == "M0")]["att"].mean()
    naive_s = f"{naive:,.0f}".replace(",", "\u202f")
    ax.set_xlabel(f"{t('ATT, USD (mean over seeds; 95 % CI). Naive difference ')}{naive_s} \\${t(' is off the axis')}")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=2, label=t(FAMILY_LABEL[f])) for f in ("matching", "weighting", "doubly_robust", "cate")]
    handles[2].set_label(t("doubly robust / balancing"))
    handles += [Line2D([], [], marker=m, color="#52514e", ls="", label=l) for m, l in (("o", "LR"), ("^", "GBM"), ("s", "RF"), ("D", t("LR, DW spec.")))]
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
    ax.set_ylabel(t("estimated ATT − 4 (across replications)"))
    ax.set_title(t("IHDP: bias across replications; numbers above boxes — 95 % CI coverage"), fontsize=9, loc="left", pad=16)
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
    POS = {0.1: 0, 0.2: 1, 0.5: 2, 1.0: 3, None: 4}  # equidistant positions; "no caliper" on the right
    r["cal_x"] = r["cal"].map(lambda c: POS[None] if c is None or (isinstance(c, float) and np.isnan(c)) else POS[c])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    a1, ak, ab, atr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    # (a) caliper × replacement; below the "without replacement" points — share of treated units that found a match
    for repl, col, lab in ((True, "#2a78d6", t("with replacement")), (False, "#eb6834", t("without replacement"))):
        g = r[r.repl == repl].groupby("cal_x").agg(att=("att", "mean"), lo=("ci_low", "mean"), hi=("ci_high", "mean"),
                                                    frac=("frac_treated_used", "mean")).sort_index()
        if g.empty:
            continue
        a1.plot(g.index, g.att, "-o", color=col, lw=2, ms=5, label=lab)
        a1.fill_between(g.index, g.lo, g.hi, color=col, alpha=0.12)
        if not repl:
            for x, (att, fr) in zip(g.index, zip(g.att, g.frac)):
                a1.annotate(f"{fr:.0%}", (x, att), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7, color=col)
    a1.axhline(benchmark, color="#0b0b0b", ls="--", lw=1); a1.set_ylabel(t("ATT, USD")); a1.legend(frameon=False, fontsize=8, loc="lower left")
    a1.set_xticks([0, 1, 2, 3, 4]); a1.set_xticklabels(["0.1", "0.2", "0.5", "1.0", t("none")])
    a1.set_xlabel(t("caliper, in SD of logit e(X)")); a1.set_title(t("(a) caliper and replacement, PSM 1:1"), fontsize=9, loc="left")

    def _series(ax, ids: list, key: str, default, xlabel: str, title: str, fmt=lambda v: f"{v:g}"):
        """Panel of ATT against one parameter: configurations ids (core + S-*), parameter value taken from params."""
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
        ax.set_xticks(xs); ax.set_xticklabels([fmt(v) for v in g.index]); ax.set_xlabel(xlabel); ax.set_ylabel(t("ATT, USD"))
        ax.set_title(title, fontsize=9, loc="left")

    _series(ak, ["M1", "M3", "S-M1-k5", "S-M1-k10"], "k", 1, t("number of matches k"),
            t("(b) number of matches, PSM k:1 with replacement"), fmt=lambda v: f"{int(v)}")
    _series(ab, ["M4", "S-M4-bw0.02", "S-M4-bw0.1", "S-M4-bw0.2"], "bandwidth", 0.06, t("bandwidth h"),
            t("(c) bandwidth, kernel matching"))
    _series(atr, ["M6", "S-M6-trim0.0", "S-M6-trim0.05"], "trim", 0.01, t("trimming threshold (weight 0 if e(X) > 1 − trim)"),
            t("(d) trimming, IPW"))
    fig.tight_layout()
    savefig(fig, out, f"F5_sensitivity_{dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lang", default="en", choices=["en", "uk"], help="language of the labels")
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    res = Path(a.results)
    out = Path(a.out or res / ("figures_uk" if a.lang == "uk" else "figures")); out.mkdir(parents=True, exist_ok=True)
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
