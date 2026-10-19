"""Deterministic fixtures for B11 (3PL/4PL diagnostics). Run from the repository root."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

OUT = Path("tests/fixtures/irt_b11")


def irf(theta, a, b, c, d):
    return c + (d - c) * expit(a * (theta[:, None] - b))


def items(rng, k, model, c_lo=.10, c_hi=.25, b_lo=-1.2, b_hi=2.0):
    a = np.round(np.linspace(.9, 2.0, k)[rng.permutation(k)], 3)
    b = np.round(np.linspace(b_lo, b_hi, k), 3)
    c = np.round(np.linspace(c_lo, c_hi, k)[rng.permutation(k)], 3)
    d = np.round(np.linspace(.88, .98, k)[rng.permutation(k)], 3) if model == "4PL" else np.ones(k)
    return a, b, c, d


def save(name, x, truth, groups=None):
    df = pd.DataFrame(x, columns=[f"M{i+1}" for i in range(x.shape[1])])
    if groups is not None:
        df.insert(0, "Grup", groups)
    df.to_csv(OUT / f"data_{name}.csv", index=False, na_rep="NA")
    truth.to_csv(OUT / f"truth_{name}.csv", index=False)


def single(name, seed, n, k, model, missing=0., testlet=None, c_override=None, hard=False, testlet_sd=1.2):
    rng = np.random.default_rng(seed)
    a, b, c, d = items(rng, k, model, *((.15, .30, -.4, 2.2) if hard else ()))
    if c_override:
        for i, v in c_override.items():
            c[i] = v
    theta = rng.normal(0, 1, n)
    eta = a * (theta[:, None] - b)
    if testlet:
        u = rng.normal(0, 1, n)
        for i in testlet:
            eta[:, i] += testlet_sd * u
    x = (rng.random((n, k)) < c + (d - c) * expit(eta)).astype(float)
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    save(name, x, pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)], "a": a, "b": b, "c": c, "d": d}))


def two_group(name, seed, model, n_r, n_f, mu_f, sd_f, dif):
    rng = np.random.default_rng(seed); k = 12
    a, b, c, d = items(rng, k, model, .15, .30, -.4, 2.2)
    xs, groups, truth = [], [], {}
    for label, n, mu, sd in (("R", n_r, 0., 1.), ("F", n_f, mu_f, sd_f)):
        p = {"a": a.copy(), "b": b.copy(), "c": c.copy(), "d": d.copy()}
        if label == "F":
            for i, change in dif.items():
                for key, v in change.items():
                    p[key][i] += v
        truth.update({f"{key}_{label}": p[key] for key in "abcd"})
        xs.append((rng.random((n, k)) < irf(rng.normal(mu, sd, n), p["a"], p["b"], p["c"], p["d"])).astype(float))
        groups += [label] * n
    save(name, np.vstack(xs), pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)], **truth}), groups)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    # Phase A: 3PL
    single("irt3pl_ok", 4101, 3000, 12, "3PL", hard=True)        # all c interior (regular case)
    single("irt3pl_easy", 4108, 2000, 12, "3PL")                 # easy items: c on the bound
    single("irt3pl_ld", 4102, 2000, 12, "3PL", testlet=(4, 5), hard=True)
    single("irt3pl_ld2", 4109, 2000, 12, "3PL", testlet=(4, 5), hard=True, testlet_sd=2.5)   # detection test only
    single("irt3pl_missing", 4103, 2000, 12, "3PL", missing=.08, hard=True)
    single("irt3pl_c0", 4104, 1500, 10, "3PL", c_override={0: 0., 1: 0., 2: .01})
    single("irt3pl_weak", 4105, 300, 8, "3PL")
    two_group("dif3pl", 4106, "3PL", 1500, 1000, -.3, 1.1, {2: {"b": .6}, 7: {"c": .15}, 9: {"a": -.9}})
    two_group("dif3pl_none", 4107, "3PL", 1500, 1000, -.3, 1.1, {})
    # Phase B: 4PL
    single("irt4pl_ok", 4201, 3000, 12, "4PL", hard=True)
    single("irt4pl_ld", 4202, 3000, 12, "4PL", testlet=(4, 5), hard=True)
    two_group("dif4pl", 4203, "4PL", 2000, 1500, -.3, 1.1, {2: {"b": .6}, 7: {"d": -.2}})
    two_group("dif4pl_none", 4204, "4PL", 2000, 1500, -.3, 1.1, {})
    print("fixtures written to", OUT)
