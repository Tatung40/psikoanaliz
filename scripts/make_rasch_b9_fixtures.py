"""Deterministic B9 fixtures: DIF, no-DIF control, local dependence (plus B8 fixtures).

dif_known: k=12, reference N=900 (theta ~ N(0,1)), focal N=500 (theta ~ N(-0.5, 1.2^2));
           uniform DIF on M3 (+0.6 logits, harder for focal) and M8 (-0.5); 8% MCAR missing.
dif_none:  same design and latent impact, no DIF (control for impact-vs-DIF confusion).
ld_pair:   k=10, N=1000, pure Rasch except a testlet effect shared by M2 and M3
           (nuisance N(0, 1) added to both logits) -> deliberate local dependence.
Run from the repository root: python scripts/make_rasch_b9_fixtures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

OUT = Path("tests/fixtures/rasch_b9")


def dif_case(dif, seed):
    rng = np.random.default_rng(seed)
    k = 12
    b = np.linspace(-1.8, 1.8, k)
    shift = np.zeros(k)
    if dif:
        shift[2], shift[7] = .6, -.5
    rows, groups = [], []
    for label, n, mu, sd, extra in (("R", 900, 0., 1., 0.), ("F", 500, -.5, 1.2, 1.)):
        theta = rng.normal(mu, sd, n)
        p = expit(theta[:, None] - (b + extra * shift)[None, :])
        rows.append((rng.random(p.shape) < p).astype(float)); groups += [label] * n
    x = np.vstack(rows)
    x[rng.random(x.shape) < .08] = np.nan
    data = pd.DataFrame(x, columns=[f"M{i+1}" for i in range(k)])
    data.insert(0, "Grup", groups)
    truth = pd.DataFrame({"Madde": data.columns[1:], "b_R": b, "b_F": b + shift, "DIF": shift})
    return data, truth


def ld_case(seed):
    rng = np.random.default_rng(seed)
    k, n = 10, 1000
    b = np.linspace(-1.6, 1.6, k)
    theta = rng.normal(0., 1., n)
    eta = theta[:, None] - b[None, :]
    nuisance = rng.normal(0., 1., n)
    eta[:, 1] += nuisance; eta[:, 2] += nuisance
    x = (rng.random((n, k)) < expit(eta)).astype(float)
    return pd.DataFrame(x, columns=[f"M{i+1}" for i in range(k)]), pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)], "b": b})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for case, (data, truth) in {"dif_known": dif_case(True, 3101), "dif_none": dif_case(False, 3102),
                                "ld_pair": ld_case(3103)}.items():
        data.to_csv(OUT / f"data_{case}.csv", index=False, na_rep="NA")
        truth.to_csv(OUT / f"truth_{case}.csv", index=False)


if __name__ == "__main__":
    main()
