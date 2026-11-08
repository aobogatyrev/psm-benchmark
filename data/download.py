"""
download.py — відтворюване завантаження даних для експерименту
«Порівняння методів оцінювання причинного ефекту» (стаття на основі КР Тімрової Н.).

Запуск (з каталогу psm-benchmark/):
    python data/download.py            # завантажити, перевірити md5, побудувати робочі файли
    python data/download.py --check    # лише перевірити наявні файли й вивести контрольні статистики

Залежності: pandas, numpy, pyreadr (для .rda), requests не потрібен (urllib).

Джерела
-------
LaLonde (версія Dehejia & Wahba 1999): R-пакет jjchern/lalonde — точні копії файлів
    nswre74_treated.txt / nswre74_control.txt / psid_controls.txt / cps_controls.txt
    з https://users.nber.org/~rdehejia/data/ (сам NBER із хмарного середовища недоступний).
IHDP (Hill 2011, setting B, реплікації 1–100): ihdp_npci_1-100.{train,test}.npz —
    копія файлів Shalit, Johansson & Sontag (2017), дзеркало в репозиторії Osier-Yi/SITE.
    Train (672) + test (75) кожної реплікації об'єднуються назад у повну вибірку 747.

Результат
---------
data/raw/      — оригінальні файли як завантажені (md5 перевіряється)
data/lalonde/  — CSV: nsw_dw.csv (445), psid_controls.csv (2490), cps_controls.csv (15992),
                 nsw.csv (722, оригінальна вибірка LaLonde 1986 без re74);
                 робочі: lalonde_dw_psid.csv (185 + 2490), lalonde_dw_cps.csv (185 + 15992),
                 lalonde_nsw_exp.csv (185 + 260, експериментальний еталон)
data/ihdp/     — ihdp_npci_1-100.csv.gz (74 700 рядків: rep, idx, split, treat, y_factual,
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

# Контрольні значення з табл. 1 Dehejia & Wahba (1999), AER/JASA; допуск 0.5 %
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
        print(f"  ok   {name} (вже є, md5 збігається)")
        return dest
    print(f"  get  {name} ← {url}")
    RAW.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    got = md5(dest)
    if got != expected_md5:
        sys.exit(f"md5 не збігається для {name}: очікувано {expected_md5}, отримано {got}")
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
    print("\nПеревірка LaLonde (еталон: Dehejia & Wahba 1999, табл. 1)")
    dw = pd.read_csv(LAL / "nsw_dw.csv")
    e = EXPECTED["nsw_dw"]
    t, c = dw[dw.treat == 1].re78.mean(), dw[dw.treat == 0].re78.mean()
    ok &= len(dw) == e["n"] and (dw.treat == 1).sum() == e["treated"] and close(t, e["re78_treated"]) and close(c, e["re78_control"])
    print(f"  nsw_dw: n={len(dw)}, treated={(dw.treat == 1).sum()}, re78 T={t:.2f} C={c:.2f}, експериментальний ATT={t - c:.2f}  (очікувано 1794.34)")
    for key in ("psid_controls", "cps_controls"):
        d = pd.read_csv(LAL / f"{key}.csv"); e = EXPECTED[key]
        good = len(d) == e["n"] and close(d.age.mean(), e["age"]) and close(d.re74.mean(), e["re74"]) and close(d.re78.mean(), e["re78"])
        ok &= good
        print(f"  {key}: n={len(d)}, age={d.age.mean():.2f}, re74={d.re74.mean():.2f}, re78={d.re78.mean():.2f}  {'ok' if good else 'НЕ ЗБІГАЄТЬСЯ'}")
    for name in ("lalonde_dw_psid", "lalonde_dw_cps", "lalonde_nsw_exp"):
        d = pd.read_csv(LAL / f"{name}.csv")
        naive = d[d.treat == 1].re78.mean() - d[d.treat == 0].re78.mean()
        print(f"  {name}: n={len(d)} ({(d.treat == 1).sum()}/{(d.treat == 0).sum()}), наївна різниця re78 = {naive:.0f} $")

    print("\nПеревірка IHDP (Hill 2011, setting B; істинний ATT = 4 за побудовою)")
    i = pd.read_csv(IHDP / "ihdp_npci_1-100.csv.gz")
    g = i.groupby("rep")
    att = g.apply(lambda d: (d.mu1 - d.mu0)[d.treat == 1].mean())
    ate = g.apply(lambda d: (d.mu1 - d.mu0).mean())
    e = EXPECTED_IHDP
    good = g.ngroups == e["reps"] and (g.size() == e["n_per_rep"]).all() and (g.treat.sum() == e["treated_per_rep"]).all() and np.allclose(att, e["true_att"], atol=1e-6)
    ok &= good
    print(f"  реплікацій={g.ngroups}, n/реплікацію={g.size().unique()}, treated/реплікацію={g.treat.sum().unique()}")
    print(f"  істинний ATT: {att.mean():.3f} (sd {att.std():.3f}); істинний ATE: {ate.mean():.3f} (sd {ate.std():.3f})  {'ok' if good else 'НЕ ЗБІГАЄТЬСЯ'}")
    print("\nУСЕ ГАРАЗД" if ok else "\nЄ РОЗБІЖНОСТІ — дані не використовувати")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="лише перевірити наявні файли")
    args = ap.parse_args()
    if not args.check:
        print("Завантаження оригінальних файлів у data/raw/")
        for name, (url, h) in RAW_FILES.items():
            fetch(name, url, h)
        print("Побудова робочих файлів")
        build_lalonde()
        build_ihdp()
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
