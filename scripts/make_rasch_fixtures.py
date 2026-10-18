"""Deterministic dichotomous Rasch fixtures for the B8 mirt / RaschPy comparisons.

rasch_main:    k=10, N=1000, theta ~ N(0, 1.2^2); items 3 and 8 deliberately misfit
               (generated with slopes 0.4 and 2.5) so that fit statistics have signal.
rasch_missing: k=10, N=800, pure Rasch, theta ~ N(0, 1), 12% MCAR missing cells.
Run from the repository root: python scripts/make_rasch_fixtures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

OUT = Path("tests/fixtures/rasch")


def make(n, k, sd, slopes, missing, seed):
    rng = np.random.default_rng(seed)
    b = np.linspace(-2., 2., k)
    theta = rng.normal(0., sd, n)
    x = (rng.random((n, k)) < expit(np.asarray(slopes) * (theta[:, None] - b))).astype(float)
    if missing:
        x[rng.random((n, k)) < missing] = np.nan
    cols = [f"M{i+1}" for i in range(k)]
    truth = pd.DataFrame({"Madde": cols, "b": b, "egim": slopes})
    return pd.DataFrame(x, columns=cols), truth, sd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    slopes = np.ones(10); slopes[2], slopes[7] = .4, 2.5
    cases = {"rasch_main": make(1000, 10, 1.2, slopes, 0., 2026),
             "rasch_missing": make(800, 10, 1.0, np.ones(10), .12, 2027)}
    rows = []
    for case, (data, truth, sd) in cases.items():
        data.to_csv(OUT / f"data_{case}.csv", index=False, na_rep="NA")
        truth.to_csv(OUT / f"truth_{case}.csv", index=False)
        rows.append({"case": case, "n": len(data), "k": data.shape[1], "sd": sd,
                     "missing_rate": float(data.isna().mean().mean())})
    pd.DataFrame(rows).to_csv(OUT / "cases.csv", index=False)
    x, w = hermgauss(41)          # core.rasch default quadrature, shared with R mirt
    pd.DataFrame({"theta": np.sqrt(2.) * x, "weight": w / np.sqrt(np.pi)}).to_csv(
        OUT / "gh41.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
