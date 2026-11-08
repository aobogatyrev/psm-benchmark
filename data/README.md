# Дані експерименту

Усе в цій теці відтворюється командою `python data/download.py` (з кореня `psm-benchmark/`);
`python data/download.py --check` лише перевіряє наявні файли. Завантажено й перевірено 2026-10-15.

## LaLonde (версія Dehejia & Wahba, 1999)

Джерело: R-пакет [jjchern/lalonde](https://github.com/jjchern/lalonde) — точні копії файлів
`nswre74_treated.txt`, `nswre74_control.txt`, `psid_controls.txt`, `cps_controls.txt`
з [users.nber.org/~rdehejia/data](https://users.nber.org/~rdehejia/data/nswdata2.html).
Назви колонок приведено до вжитку MatchIt: `education → educ`, `hispanic → hispan`.
У робочих файлах додано індикатори нульового доходу `u74`, `u75` (специфікація DW).

| Файл | N (treated / control) | Наївна різниця re78 | Призначення |
| --- | --- | --- | --- |
| `lalonde/lalonde_dw_psid.csv` | 185 / 2 490 | −15 205 $ | **основний** спостережний бенчмарк |
| `lalonde/lalonde_dw_cps.csv` | 185 / 15 992 | −8 498 $ | чутливість до контрольної групи |
| `lalonde/lalonde_nsw_exp.csv` | 185 / 260 | +1 794 $ | експериментальний еталон ATT = 1 794.34 $ |
| `lalonde/nsw_dw.csv` | 445 (DW-підвибірка NSW) | | джерело для двох файлів вище |
| `lalonde/psid_controls.csv`, `lalonde/cps_controls.csv` | 2 490; 15 992 | | контрольні групи |
| `lalonde/nsw.csv` | 722 (оригінал LaLonde 1986, без re74) | | не використовується в статті |

Контрольні значення (табл. 1 DW 1999), які перевіряє скрипт: NSW treated re78 = 6 349.14, NSW control re78 = 4 554.80;
PSID: age 34.85, re74 19 428.75, re78 21 553.92; CPS: age 33.23, re74 14 016.80, re78 14 846.66.

Колонки робочих файлів: `treat, age, educ, black, hispan, married, nodegree, re74, re75, u74, u75, re78`.

## IHDP (Hill 2011, setting B, реплікації 1–100)

Джерело: `ihdp_npci_1-100.train.npz` і `ihdp_npci_1-100.test.npz` з [Osier-Yi/SITE](https://github.com/Osier-Yi/SITE/tree/master/data) —
копія файлів Shalit, Johansson & Sontag (2017), які використовують CFRNet, CEVAE, Dragonnet тощо
(первинне джерело fredjo.com недоступне з хмарного середовища). Train (672) + test (75) кожної
реплікації об'єднано у повну вибірку 747; колонка `split` зберігає початковий поділ.

`ihdp/ihdp_npci_1-100.csv.gz` — 74 700 рядків: `rep` (1–100), `idx`, `split`, `treat`, `y_factual`, `y_cfactual`,
`mu0`, `mu1`, `x1`–`x25` (x1–x6 неперервні, стандартизовані; x7–x25 бінарні).
У кожній реплікації 747 спостережень, 139 treated.

Істинні ефекти: **ATT = 4.000 у кожній реплікації** (так побудовано setting B у Hill 2011 — зсув ω підібрано
під CATT = 4), ATE = 4.52 ± 1.68 по реплікаціях, SD індивідуального ефекту всередині реплікації ≈ 5.6,
наївна різниця 4.67 ± 2.13. Отже, на IHDP зміщення ATT рахується відносно 4; істинний ATT для
довільної підвибірки — `mean(mu1 − mu0 | treat = 1)`.

## Контрольні суми

Оригінали (`data/raw/`): `nsw_dw.rda` 793e84cce2629d06d850aa2bdf2ce40e · `psid_controls.rda` 312c31a2c53eaeafc6c7517ee7e85a42 ·
`cps_controls.rda` 3c1190de1bb476bc50d899e4f69e2745 · `nsw.rda` 1886db3c0965beb8e12633232e9dcbfe ·
`ihdp_npci_1-100.train.npz` 397574f41c0f2f0afeb8c91d29289ac9 · `ihdp_npci_1-100.test.npz` 8b1f1e63cb05102ed1d8257b0cba9544.

Похідні: `lalonde_dw_psid.csv` d03567eea7e8961a457f11934c9f409d · `lalonde_dw_cps.csv` d4ae31d3f04478df885b0d57964c9f92 ·
`lalonde_nsw_exp.csv` 3ac4ee35b767655c27b564183f7881f9 · `nsw_dw.csv` f6ad576ad91a07619f574f7abc063c31 ·
`psid_controls.csv` f4a95584507f57f8132cc0b1e50e7591 · `cps_controls.csv` 49d87f2ed190bea062823856b0772034 ·
`ihdp_npci_1-100.csv.gz` 38c36dae28029deb8bf57155cf41214a (md5 gz-файлу може відрізнятися між версіями pandas; звіряти статистики через `--check`).

## Що НЕ використовувати

`section_5_experiments/data/lalonde.csv` та `ihdp.csv` з архіву КР — синтетичні демонстраційні файли
(111/503 treated/control, re74 контролю 17 666 $; 345 treated в IHDP). Жодна цифра, отримана на них, у статтю не йде.
