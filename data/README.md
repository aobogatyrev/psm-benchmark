# Data

Everything in this folder is reproduced by `python data/download.py` (run from the root of `psm-benchmark/`);
`python data/download.py --check` only verifies the files that are already present. Downloaded and verified on 2026-10-15.
The raw and processed files are not tracked in git (see `.gitignore`).

## LaLonde (Dehejia & Wahba, 1999 version)

Source: the R package [jjchern/lalonde](https://github.com/jjchern/lalonde) — exact copies of the files
`nswre74_treated.txt`, `nswre74_control.txt`, `psid_controls.txt`, `cps_controls.txt`
from [users.nber.org/~rdehejia/data](https://users.nber.org/~rdehejia/data/nswdata2.html).
Column names follow the MatchIt convention: `education → educ`, `hispanic → hispan`.
The working files add the zero-earnings indicators `u74`, `u75` (Dehejia–Wahba specification).

| File | N (treated / control) | Naive difference in re78 | Purpose |
| --- | --- | --- | --- |
| `lalonde/lalonde_dw_psid.csv` | 185 / 2,490 | −15,205 USD | **main** observational benchmark |
| `lalonde/lalonde_dw_cps.csv` | 185 / 15,992 | −8,498 USD | sensitivity to the control sample |
| `lalonde/lalonde_nsw_exp.csv` | 185 / 260 | +1,794 USD | experimental benchmark, ATT = 1,794.34 USD |
| `lalonde/nsw_dw.csv` | 445 (DW subsample of NSW) | | source of the two files above |
| `lalonde/psid_controls.csv`, `lalonde/cps_controls.csv` | 2,490; 15,992 | | control samples |
| `lalonde/nsw.csv` | 722 (original LaLonde 1986 sample, no re74) | | not used in the article |

Reference values (Table 1 of DW 1999) checked by the script: NSW treated re78 = 6,349.14, NSW control re78 = 4,554.80;
PSID: age 34.85, re74 19,428.75, re78 21,553.92; CPS: age 33.23, re74 14,016.80, re78 14,846.66.

Columns of the working files: `treat, age, educ, black, hispan, married, nodegree, re74, re75, u74, u75, re78`.

## IHDP (Hill 2011, setting B, replications 1–100)

Source: `ihdp_npci_1-100.train.npz` and `ihdp_npci_1-100.test.npz` from [Osier-Yi/SITE](https://github.com/Osier-Yi/SITE/tree/master/data) —
a copy of the files of Shalit, Johansson & Sontag (2017) used by CFRNet, CEVAE, Dragonnet and others
(the primary source fredjo.com was not reachable from the build environment). Train (672) + test (75) of every
replication are merged back into the full sample of 747; the column `split` keeps the original split.

`ihdp/ihdp_npci_1-100.csv.gz` — 74,700 rows: `rep` (1–100), `idx`, `split`, `treat`, `y_factual`, `y_cfactual`,
`mu0`, `mu1`, `x1`–`x25` (x1–x6 continuous, standardised; x7–x25 binary).
Every replication has 747 observations, 139 of them treated.

True effects: **ATT = 4.000 in every replication** (by construction of setting B in Hill 2011 — the offset ω is chosen
so that CATT = 4), ATE = 4.52 ± 1.68 across replications, SD of the individual effect within a replication ≈ 5.6,
naive difference 4.67 ± 2.13. The bias of the ATT on IHDP is therefore measured against 4; the true ATT of an
arbitrary subsample is `mean(mu1 − mu0 | treat = 1)`.

## Checksums

Raw files (`data/raw/`): `nsw_dw.rda` 793e84cce2629d06d850aa2bdf2ce40e · `psid_controls.rda` 312c31a2c53eaeafc6c7517ee7e85a42 ·
`cps_controls.rda` 3c1190de1bb476bc50d899e4f69e2745 · `nsw.rda` 1886db3c0965beb8e12633232e9dcbfe ·
`ihdp_npci_1-100.train.npz` 397574f41c0f2f0afeb8c91d29289ac9 · `ihdp_npci_1-100.test.npz` 8b1f1e63cb05102ed1d8257b0cba9544.

Derived files: `lalonde_dw_psid.csv` d03567eea7e8961a457f11934c9f409d · `lalonde_dw_cps.csv` d4ae31d3f04478df885b0d57964c9f92 ·
`lalonde_nsw_exp.csv` 3ac4ee35b767655c27b564183f7881f9 · `nsw_dw.csv` f6ad576ad91a07619f574f7abc063c31 ·
`psid_controls.csv` f4a95584507f57f8132cc0b1e50e7591 · `cps_controls.csv` 49d87f2ed190bea062823856b0772034 ·
`ihdp_npci_1-100.csv.gz` 38c36dae28029deb8bf57155cf41214a (the md5 of the gz file may differ between pandas versions; verify the statistics with `--check`).
