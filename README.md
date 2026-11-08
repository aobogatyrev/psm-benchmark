# psm-benchmark

Code, result files and figures for the article

> Timrova N., Lobyntsev A., Bogatyrev O. **Does machine learning improve propensity score matching? A controlled comparison of classical and learning-based estimators of the treatment effect on the treated.** *Bulletin of Cherkasy University. Series “Applied Mathematics. Informatics”*, 2026 (submitted).

Nineteen configurations of matching, weighting, balancing and learning-based estimators of the ATT are
compared in one implementation, in which every matching/weighting/balancing estimator returns a weight
vector, so that balance, effective sample size and the bootstrap are computed by shared code. Benchmarks:
LaLonde–Dehejia–Wahba (PSID-1 and CPS-1 controls; experimental benchmark 1,794 USD) and IHDP,
replications 1–100 (true ATT = 4.000).

Every number in the article comes from `results/final/runs.csv` (one row per dataset × replication ×
configuration × seed) and `results/final/balance.csv` (per-covariate SMD before/after); the tables and
figures are produced from these files by `make_tables.py` and `make_figures.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # + pip install doubleml econml for X1–X4
python data/download.py                      # downloads the public data and checks them (see data/README.md)
python -m pytest -q tests/                   # 7 smoke tests, ~30 s
python run_experiment.py --dataset lalonde_nsw_exp --configs core --seeds 1 --bootstrap 30   # ~3 min
```

Full run used in the article (≈ 12–14 h on 4 cores; resumable, step by step):

```bash
bash run_full.sh                              # → results/final/runs.csv, balance.csv
python make_tables.py  --results results/final
python make_figures.py --results results/final            # add --lang uk for Ukrainian labels
```

## Layout

```text
psm-benchmark/
  data/download.py        download + md5 + checks against Table 1 of Dehejia & Wahba (1999) and IHDP properties
  data/README.md          data sources and file descriptions (raw and processed files are not tracked; reproducible)
  psmbench/
    data.py               load_dataset(name, rep) → (DataFrame, DatasetSpec)
    propensity.py         propensity models: lr, lr_dw (Dehejia–Wahba specification), gbm (out-of-fold), rf (OOB + isotonic)
    matching.py           nn_match (k:1, caliper in SD of the logit, with/without replacement), kernel_match, mahalanobis_match
    weighting.py          ipw_att_weights (Hájek, trimming), entropy_balancing (dual convex problem, L-BFGS-B)
    estimators.py         estimate(df, spec, cfg, seed, B) — single interface; AIPW; everything through a weight vector
    inference.py          Abadie–Imbens (2006) SE for nearest-neighbour matching; stratified bootstrap
    balance.py            SMD before/after (pooled pre-balancing SD in the denominator), ESS
    extended.py           X1 DML-IRM (doubleml), X2 cross-fitted AIPW, X3 X-learner, X4 causal forest (econml)
    configs.py            CORE (M0–M14, M1+), EXTENDED (X1–X4), SENSITIVITY (S-*)
  run_experiment.py       CLI → results/<out>/runs.csv (+ balance.csv)
  run_full.sh             the full protocol of the article as resumable steps
  make_tables.py          T1–T5 (CSV + Markdown) from runs.csv / balance.csv
  make_figures.py         F1–F5 (PNG + PDF); English labels by default, --lang uk for Ukrainian
  tests/test_smoke.py
  results/final/          runs.csv, balance.csv, tables/, figures/ (en), figures_uk/ (uk)
```

## Configurations

| ID | Estimator | e(X) | Settings | SE |
| --- | --- | --- | --- | --- |
| M0 | naive difference | — | — | analytic |
| M1 | PSM 1:1 NN, with replacement | LR | caliper 0.2·SD(logit) | Abadie–Imbens |
| M2 | PSM 1:1 NN, without replacement | LR | caliper 0.2·SD, random order | bootstrap |
| M3 | PSM 3:1 NN | LR | caliper 0.2·SD | Abadie–Imbens |
| M4 | kernel matching | LR | Epanechnikov, h = 0.06, common support | bootstrap |
| M5 | Mahalanobis 1:1 NN | — | standardised X | Abadie–Imbens |
| M6 | IPW (ATT) | LR | normalised, trim ê > 0.99 | bootstrap |
| M7 | AIPW | LR | linear outcome model | bootstrap |
| M8 | entropy balancing | — | 1st moments of all + 2nd of continuous covariates | bootstrap |
| M9–M11 | M1, M4, M6 | GBM (out-of-fold) | | |
| M12–M14 | M1, M4, M6 | RF (OOB, isotonic) | | |
| M1+ | PSM 1:1 NN | LR, Dehejia–Wahba specification | LaLonde only | Abadie–Imbens |
| X1 | DML-IRM, ATTE score (doubleml) | GBM nuisances | 5-fold cross-fitting | analytic (influence function) |
| X2 | AIPW, cross-fitted | GBM nuisances | 5-fold cross-fitting | bootstrap |
| X3 | X-learner (Künzel et al., 2019) | GBM | ATT = mean τ̂(X) over treated | bootstrap |
| X4 | causal forest (econml CausalForestDML) | GBM orthogonalisation | 1,000 honest trees | analytic |

Sensitivity grid (`--configs sensitivity`): caliper {none, 0.1, 0.2, 0.5, 1.0} × replacement, k ∈ {1, 3, 5, 10},
kernel bandwidth {0.02, 0.06, 0.10, 0.20}, IPW trimming {none, 0.01, 0.05}; core set repeated with CPS-1 controls.

## Output columns (`runs.csv`)

`dataset, rep, config_id, seed, family, estimator, ps_model, params, n_treated, n_control, n_treated_used,
frac_treated_used, ess_control, naive, att, se, se_method, ci_low, ci_high, true_att, bias, covered, pehe,
mean/max_abs_smd_before, mean/max_abs_smd_after, frac_smd_lt_01_after, runtime_s, ps_auc, ps_logloss, ps_brier,
ps_frac_e_gt_095, ci_pct_low/high, bootstrap_valid, eb_converged, eb_n_moments, eb_max_moment_imbalance`.
Re-running with the same keys replaces the rows.

## Data

All data are public and are downloaded by `data/download.py` from the original sources: the Dehejia–Wahba
LaLonde files (NSW, PSID-1, CPS-1) and the IHDP replications 1–100 of Shalit et al. (2017) / Hill (2011).
The script verifies file checksums and the reference statistics of Table 1 in Dehejia & Wahba (1999).

## Citation

See `CITATION.cff`. Please cite the article above and, for the software, this repository.

## License

MIT (see `LICENSE`). The data remain under the terms of their original sources.
