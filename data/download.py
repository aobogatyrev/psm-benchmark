"""
download.py — reproducible download of the data for the experiment of the article
"Does machine learning improve propensity score matching?" (Timrova, Lobyntsev, Bogatyrev).

Usage (from the psm-benchmark/ directory):
    python data/download.py            # download, verify md5, build the working files
    python data/download.py --check    # only verify the existing files and print the reference statistics

Dependencies: pandas, numpy, pyreadr (for .rda); requests is not needed (urllib).

Sources
-------
LaLonde (Dehejia & Wahba 1999 version): the R package jjchern/lalonde — exact copies of the files
    nswre74_treated.txt / nswre74_control.txt / psid_controls.txt / cps_controls.txt
    from https://users.nber.org/~rdehejia/data/ (the NBER site itself was not reachable from the build environment).
IHDP (Hill 2011, setting B, replications 1–100): ihdp_npci_1-100.{train,test}.npz —
    a copy of the files of Shalit, Johansson & Sontag (2017), mirrored in the Osier-Yi/SITE repository.
    Train (672) + test (75) of every replication are merged back into the full sample of 747.

Output
------
data/raw/      — the original files as downloaded (md5 verified)
data/lalonde/  — CSV: nsw_dw.csv (445), psid_controls.csv (2490), cps_controls.csv (15992),
                 nsw.csv (722, the original LaLonde 1986 sample without re74);
                 working files: lalonde_dw_psid.csv (185 + 2490), lalonde_dw_cps.csv (185 + 15992),
                 lalonde_nsw_exp.csv (185 + 260, experimental benchmark)
data/ihdp/     — ihdp_npci_1-100.csv.gz (74,700 rows: rep, idx, split, treat, y_factual,
                 y_cfactual, mu0, mu1, x1..x25)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
LAL = HERE / "lalonde"
IHDP = HERE / "ihdp"

RAW_FILES = {
    # name: (url, md5)
    "nsw_dw.rda": ("https://raw.githubusercontent.com/jjchern/lalonde/master/data/nsw_dw.rda",
                   "793e84cce2629d06d850aa2bdf2ce40e"),
    "psid_controls.rda": ("https://raw.githubusercontent.com/jjchern/lalonde/master/data/psid_controls.rda",
                          "312c31a2c53eaeafc6c7517ee7e85a42"),
    "cps_controls.rda": ("https://raw.githubusercontent.com/jjchern/lalonde/master/data/cps_controls.rda",
                         "3c1190de1bb476bc50d899e4f69e2745"),
    "nsw.rda": ("https://raw.githubusercontent.com/jjchern/lalonde/master/data/nsw.rda",
                "1886db3c0965beb8e12633232e9dcbfe"),
    "ihdp_npci_1-100.train.npz": ("https://raw.githubusercontent.com/Osier-Yi/SITE/master/data/ihdp_npci_1-100.train.npz",
                                  "397574f41c0f2f0afeb8c91d29289ac9"),
    "ihdp_npci_1-100.test.npz": ("https://raw.githubusercontent.com/Osier-Yi/SITE/master/data/ihdp_npci_1-100.test.npz",
                                 "8b1f1e63cb05102ed1d8257b0cba9544"),
}

# Reference values from Table 1 of Dehejia & Wahba (1999); tolerance 0.5 %
EXPECTED = {
    "nsw_dw": {"n": 445, "treated": 185, "re78_treated": 6349.14, "re78_control": 4554.80},
    "psid_controls": {"n": 2490, "age": 34.85, "re74": 19428.75, "re78": 21553.92},
    "cps_controls": {"n": 15992, "age": 33.23, "re74": 14016.80, "re78": 14846.66},
}
EXPECTED_IHDP = {"reps": 100, "n_per_rep": 747, "treated_per_rep": 139, "true_att": 4.0}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, expected_md5: str) -> Path:
    dest = RAW / name
    if dest.exists() and md5(dest) == expected_md5:
        print(f"  ok   {name} (already present, md5 matches)")
        return dest
    print(f"  get  {name} ← {url}")
    RAW.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    got = md5(dest)
    if got != expected_md5:
        sys.exit(f"md5 mismatch for {name}: expected {expected_md5}, got {got}")
    print(f"  ok   {name} md5={got}")
    return dest


def build_lalonde() -> None:
    import pyreadr  # pip install pyreadr

    LAL.mkdir(parents=True, exist_ok=True)
    frames = {}
    for key in ("nsw_dw", "psid_controls", "cps_controls", "nsw"):
        df = next(iter(pyreadr.read_r(RAW / f"{key}.rda").values()))
        df = df.rename(columns={"education": "educ", "hispanic": "hispan"})
        df["treat"] = df["treat"].astype(int)
        df.to_csv(LAL / f"{key}.csv", index=False)
        frames[key] = df

    def with_zero_flags(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["u74"] = (df["re74"] == 0).astype(int)
        df["u75"] = (df["re75"] == 0).astype(int)
        return df

    treated = frames["nsw_dw"][frames["nsw_dw"].treat == 1]
    exp_control = frames["nsw_dw"][frames["nsw_dw"].treat == 0]
    cols = ["treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "u74", "u75", "re78"]

    with_zero_flags(pd.concat([treated, frames["psid_controls"]], ignore_index=True))[cols] \
        .to_csv(LAL / "lalonde_dw_psid.csv", index=False)
    with_zero_flags(pd.concat([treated, frames["cps_controls"]], ignore_index=True))[cols] \
        .to_csv(LAL / "lalonde_dw_cps.csv", index=False)
    with_zero_flags(pd.concat([treated, exp_control], ignore_index=True))[cols] \
        .to_csv(LAL / "lalonde_nsw_exp.csv", index=False)


def build_ihdp() -> None:
    IHDP.mkdir(parents=True, exist_ok=True)
    tr = np.load(RAW / "ihdp_npci_1-100.train.npz")
    te = np.load(RAW / "ihdp_npci_1-100.test.npz")
    frames = []
    for r in range(tr["x"].shape[2]):
        parts = []
        for src, part in ((tr, "train"), (te, "test")):
            d = pd.DataFrame(src["x"][:, :, r], columns=[f"x{i + 1}" for i in range(25)])
            for col, key in (("mu1", "mu1"), ("mu0", "mu0"), ("y_cfactual", "ycf"), ("y_factual", "yf")):
                d.insert(0, col, src[key][:, r])
            d.insert(0, "treat", src["t"][:, r].astype(int))
            d.insert(0, "split", part)
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        d.insert(0, "rep", r + 1)
        d.insert(1, "idx", range(len(d)))
        frames.append(d)
    pd.concat(frames, ignore_index=True).to_csv(
        IHDP / "ihdp_npci_1-100.csv.gz", index=False, compression="gzip", float_format="%.6g")


def close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol * abs(b)


def check() -> bool:
    ok = True
    print("\nChecking LaLonde (reference: Dehejia & Wahba 1999, Table 1)")
    dw = pd.read_csv(LAL / "nsw_dw.csv")
    e = EXPECTED["nsw_dw"]
    t, c = dw[dw.treat == 1].re78.mean(), dw[dw.treat == 0].re78.mean()
    ok &= len(dw) == e["n"] and (dw.treat == 1).sum() == e["treated"] and close(t, e["re78_treated"]) and close(c, e["re78_control"])
    print(f"  nsw_dw: n={len(dw)}, treated={(dw.treat == 1).sum()}, re78 T={t:.2f} C={c:.2f}, experimental ATT={t - c:.2f}  (expected 1794.34)")
    for key in ("psid_controls", "cps_controls"):
        d = pd.read_csv(LAL / f"{key}.csv"); e = EXPECTED[key]
        good = len(d) == e["n"] and close(d.age.mean(), e["age"]) and close(d.re74.mean(), e["re74"]) and close(d.re78.mean(), e["re78"])
        ok &= good
        print(f"  {key}: n={len(d)}, age={d.age.mean():.2f}, re74={d.re74.mean():.2f}, re78={d.re78.mean():.2f}  {'ok' if good else 'MISMATCH'}")
    for name in ("lalonde_dw_psid", "lalonde_dw_cps", "lalonde_nsw_exp"):
        d = pd.read_csv(LAL / f"{name}.csv")
        naive = d[d.treat == 1].re78.mean() - d[d.treat == 0].re78.mean()
        print(f"  {name}: n={len(d)} ({(d.treat == 1).sum()}/{(d.treat == 0).sum()}), naive difference in re78 = {naive:.0f} USD")

    print("\nChecking IHDP (Hill 2011, setting B; true ATT = 4 by construction)")
    i = pd.read_csv(IHDP / "ihdp_npci_1-100.csv.gz")
    g = i.groupby("rep")
    att = g.apply(lambda d: (d.mu1 - d.mu0)[d.treat == 1].mean())
    ate = g.apply(lambda d: (d.mu1 - d.mu0).mean())
    e = EXPECTED_IHDP
    good = g.ngroups == e["reps"] and (g.size() == e["n_per_rep"]).all() and (g.treat.sum() == e["treated_per_rep"]).all() and np.allclose(att, e["true_att"], atol=1e-6)
    ok &= good
    print(f"  replications={g.ngroups}, n per replication={g.size().unique()}, treated per replication={g.treat.sum().unique()}")
    print(f"  true ATT: {att.mean():.3f} (sd {att.std():.3f}); true ATE: {ate.mean():.3f} (sd {ate.std():.3f})  {'ok' if good else 'MISMATCH'}")
    print("\nALL CHECKS PASSED" if ok else "\nMISMATCHES FOUND — do not use the data")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify the existing files")
    args = ap.parse_args()
    if not args.check:
        print("Downloading the original files into data/raw/")
        for name, (url, h) in RAW_FILES.items():
            fetch(name, url, h)
        print("Building the working files")
        build_lalonde()
        build_ihdp()
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
