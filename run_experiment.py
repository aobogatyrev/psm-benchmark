"""run_experiment.py — single entry point for all runs.

Examples:
  python run_experiment.py --dataset lalonde_nsw_exp --configs core --seeds 1 --bootstrap 50   # quick check
  python run_experiment.py --dataset lalonde_dw_psid --configs core --seeds 10 --bootstrap 500
  python run_experiment.py --dataset ihdp --reps 1-100 --configs core --bootstrap 200 --jobs 4
  python run_experiment.py --dataset lalonde_dw_psid --configs sensitivity --seeds 10 --bootstrap 200

Every run (dataset, rep, config, seed) → one row in results/runs.csv (append; duplicates
by key are replaced) and per-covariate SMD rows in results/balance.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psmbench import estimate, get_configs, load_dataset  # noqa: E402

KEY = ["dataset", "rep", "config_id", "seed"]
SEED_BASE = 20260001


def parse_reps(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_one(dataset: str, rep, cfg: dict, seed: int, B: int):
    df, spec = load_dataset(dataset, rep=rep)
    r = estimate(df, spec, cfg, seed=seed, bootstrap_B=B)
    true = spec.true_att
    pehe = np.nan
    if r.cate is not None and {"mu0", "mu1"} <= set(df.columns):
        pehe = float(np.sqrt(np.mean((r.cate - (df["mu1"] - df["mu0"]).values) ** 2)))
    row = {
        "dataset": dataset, "rep": rep if rep is not None else 0, "config_id": cfg["id"], "seed": seed,
        "family": cfg["family"], "estimator": cfg["estimator"], "ps_model": cfg.get("ps_model") or "",
        "params": json.dumps(cfg.get("params", {}), sort_keys=True),
        "n_treated": r.n_treated, "n_control": r.n_control, "n_treated_used": r.n_treated_used,
        "frac_treated_used": r.n_treated_used / r.n_treated, "ess_control": r.ess_control,
        "naive": r.naive, "att": r.att, "se": r.se, "se_method": r.se_method,
        "ci_low": r.ci_low, "ci_high": r.ci_high,
        "true_att": true,
        "bias": (r.att - true) if true is not None else np.nan,
        "covered": int(r.ci_low <= true <= r.ci_high) if (true is not None and np.isfinite(r.se)) else np.nan,
        "pehe": pehe,
        **{f"ps_{k}": v for k, v in r.ps_metrics.items()},
        **r.balance,
        **{k: v for k, v in r.info.items()},
        "runtime_s": r.runtime_s,
    }
    bal = r.smd.copy()
    for k, v in zip(KEY, (dataset, row["rep"], cfg["id"], seed)):
        bal[k] = v
    return row, bal


def upsert(path: Path, new: pd.DataFrame, key):
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=key, keep="last")
    else:
        merged = new
    merged.to_csv(path, index=False)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reps", default="1", help="for ihdp: '1-100' or '1,2,5'")
    ap.add_argument("--configs", default="core", help="core | extended | sensitivity | all | comma-separated list of ids")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (LaLonde); for ihdp — 1 per replication")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.configs in ("core", "extended", "sensitivity", "all"):
        cfgs = get_configs(args.configs)
    else:
        ids = set(args.configs.split(","))
        cfgs = [c for c in get_configs("all") if c["id"] in ids]
    cfgs = [c for c in cfgs if "datasets" not in c or args.dataset in c["datasets"]]

    reps = parse_reps(args.reps) if args.dataset == "ihdp" else [None]
    seeds = [SEED_BASE + i for i in range(args.seeds)] if args.dataset != "ihdp" else None

    jobs = []
    for rep in reps:
        rep_seeds = seeds if seeds is not None else [SEED_BASE + rep]
        for cfg in cfgs:
            for s in rep_seeds:
                jobs.append((args.dataset, rep, cfg, s, args.bootstrap))
    print(f"{len(jobs)} runs: dataset={args.dataset}, configs={[c['id'] for c in cfgs]}, "
          f"reps={reps if reps != [None] else '-'}, bootstrap B={args.bootstrap}, jobs={args.jobs}")

    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_one)(*j) for j in jobs)
    rows = pd.DataFrame([r for r, _ in results])
    bal = pd.concat([b for _, b in results], ignore_index=True)

    out = Path(args.out); out.mkdir(exist_ok=True)
    runs = upsert(out / "runs.csv", rows, KEY)
    upsert(out / "balance.csv", bal, KEY + ["covariate"])
    print(f"done in {time.time() - t0:.0f} s; runs.csv now has {len(runs)} rows")

    show = ["config_id", "ps_model", "att", "se", "ci_low", "ci_high", "bias", "frac_treated_used",
            "ess_control", "mean_abs_smd_after", "max_abs_smd_after"]
    summary = rows.groupby(["config_id", "ps_model"], sort=False)[[c for c in show if c not in ("config_id", "ps_model")]].median(numeric_only=True)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\nMedians over runs:")
        print(summary)


if __name__ == "__main__":
    main()
