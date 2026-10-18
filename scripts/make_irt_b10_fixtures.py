"""Deterministic fixtures for B10 (1PL/2PL diagnostics). Run from the repository root."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

OUT = Path("tests/fixtures/irt_b10")


def responses(rng, theta, a, b):
    return (rng.random((len(theta), len(b))) < expit(a[None, :] * (theta[:, None] - b[None, :]))).astype(float)


def frame(x, k):
    return pd.DataFrame(x, columns=[f"M{i+1}" for i in range(k)])


def single(name, seed, n, k, model, missing=0., testlet=None):
    rng = np.random.default_rng(seed)
    b = np.linspace(-1.8, 1.8, k)
    a = np.ones(k) if model == "1PL" else np.round(np.linspace(.6, 2., k)[rng.permutation(k)], 3)
    theta = rng.normal(0, 1, n)
    eta = a[None, :] * (theta[:, None] - b[None, :])
    if testlet:
        u = rng.normal(0, 1, n)
        for i in testlet:
            eta[:, i] += 1.0 * u
    x = (rng.random((n, k)) < expit(eta)).astype(float)
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    frame(x, k).to_csv(OUT / f"data_{name}.csv", index=False, na_rep="NA")
    pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)], "a": a, "b": b}).to_csv(OUT / f"truth_{name}.csv", index=False)


def two_group(name, seed, model, n_r, n_f, mu_f, sd_f, dif, missing=0.):
    rng = np.random.default_rng(seed); k = 12
    b = np.linspace(-1.8, 1.8, k)
    a = np.ones(k) if model == "1PL" else np.round(np.linspace(.8, 1.8, k)[rng.permutation(k)], 3)
    rows, groups, truth = [], [], []
    for label, n, mu, sd in (("R", n_r, 0., 1.), ("F", n_f, mu_f, sd_f)):
        aa, bb = a.copy(), b.copy()
        if label == "F":
            for i, (da, db) in dif.items():
                aa[i] += da; bb[i] += db
        truth.append(pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)], f"a_{label}": aa, f"b_{label}": bb}))
        rows.append(responses(rng, rng.normal(mu, sd, n), aa, bb)); groups += [label] * n
    x = np.vstack(rows)
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    d = frame(x, k); d.insert(0, "Grup", groups)
    d.to_csv(OUT / f"data_{name}.csv", index=False, na_rep="NA")
    truth[0].merge(truth[1], on="Madde").to_csv(OUT / f"truth_{name}.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    single("irt1pl_ok", 3201, 1000, 10, "1PL")
    single("irt2pl_ok", 3202, 1200, 10, "2PL")
    single("irt2pl_missing", 3203, 1000, 10, "2PL", missing=.10)
    single("irt2pl_ld", 3204, 1000, 10, "2PL", testlet=(3, 4))
    two_group("dif1pl_uniform", 3205, "1PL", 1000, 600, -.4, 1.2, {2: (0., .6), 8: (0., -.5)}, missing=.06)
    two_group("dif2pl_mixed", 3206, "2PL", 1200, 800, -.3, 1.1, {2: (0., .6), 6: (-.9, 0.)})
    two_group("dif2pl_none", 3207, "2PL", 1200, 800, -.3, 1.1, {})
    print("fixtures written to", OUT)
