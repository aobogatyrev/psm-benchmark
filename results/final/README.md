# results/final

Result files of the full run reported in the article (LaLonde–DW PSID-1: 10 seeds, B = 500; sensitivity grid: B = 200;
CPS-1: 5 and 3 seeds, B ≤ 200; IHDP replications 1–100: B = 200).

- `runs.csv` — one row per dataset × replication × configuration × seed (see the column list in the root README).
- `balance.csv` — standardised mean difference of every covariate before and after each configuration (Love plots).
- `tables/` — T1 datasets, T2 balance (PSID, CPS), T3 main LaLonde (PSID, CPS), T4 main IHDP, T5 propensity-model quality;
  Table 1 of the article = `T3_main_lalonde_dw_psid`, Table 2 = `T4_main_ihdp`, the CPS-1 table = `T3_main_lalonde_dw_cps`.
- `figures_en/` — F1 overlap (Fig. 1 of the article), F2 Love plot, F3 forest plot, F4 IHDP bias/coverage (Fig. 2), F5 sensitivity (Fig. 3);
  `figures/` — the same with Ukrainian labels.

Regenerate: `python make_tables.py --results results/final` and `python make_figures.py --results results/final --lang en`.
