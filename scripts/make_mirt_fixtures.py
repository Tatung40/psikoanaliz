"""Generate deterministic 3PL/4PL fixtures for the R mirt comparison (B3).

Writes small CSV files to tests/fixtures/mirt/: response data, generating
item parameters (a, b, c, d) and the 21-node Gauss-Hermite grid for N(0,1)
(theta = sqrt(2) x, weight = w / sqrt(pi)), so that R mirt integrates on
exactly the same nodes and weights as the Python code. Run from repo root:
    python scripts/make_mirt_fixtures.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt_simulation import simulate_irt

OUT = Path("tests/fixtures/mirt")
POINTS = 21
SEED = 2026


def items(model, design):
    """B2 designs; 4PL uses k=5 because complete 0/1 data on k items have
    2**k - 1 free cell probabilities: 4PL with k=4 has 16 parameters for 15
    degrees of freedom (structurally unidentified; mirt refuses it)."""
    k = 4 if model == "3PL" else 5
    if design == "est":       # free-estimation comparison: interior 3PL MLE
        k = 8
        a, b = np.linspace(1.6, 2.4, k), np.linspace(-1.2, 1.2, k)
        c, d = np.linspace(.15, .22, k), np.ones(k)
    elif design == "weak":
        a, b = np.linspace(1., 2., k), np.linspace(-1, 1, k)
        c, d = np.linspace(.12, .2, k), np.linspace(.85, .92, k)
    else:
        a, b = np.linspace(2., 2.6, k), np.linspace(-.4, .4, k)
        c, d = np.linspace(.15, .22, k), np.linspace(.82, .88, k)
    return pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)], "a": a, "b": b,
                         "c": c, "d": np.ones(k) if model == "3PL" else d})


# B5: free 4PL estimation fixtures (written to a separate case list so the
# B3 reference run is unchanged).
FREE_4PL = [("4pl_free_k5", 5, 3000), ("4pl_free_k8", 8, 3000)]


def free_4pl_items(k):
    return pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)],
                         "a": np.linspace(1.6, 2.4, k) if k == 5 else np.linspace(1.4, 2.4, k),
                         "b": np.linspace(-1., 1., k) if k == 5 else np.linspace(-1.4, 1.4, k),
                         "c": np.linspace(.12, .20, k), "d": np.linspace(.86, .94, k)})


CASES = [("3pl_strong", "3PL", "strong", 2000), ("4pl_strong", "4PL", "strong", 2000),
         ("3pl_weak", "3PL", "weak", 400), ("4pl_weak", "4PL", "weak", 400),
         ("3pl_est", "3PL", "est", 3000)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, w = hermgauss(POINTS)
    pd.DataFrame({"theta": np.sqrt(2.) * x, "weight": w / np.sqrt(np.pi)}).to_csv(
        OUT / "gh21.csv", index=False, float_format="%.17g")
    rows = []
    for case, model, design, n in CASES:
        truth = items(model, design)
        data, _ = simulate_irt(n, truth, seed=SEED)
        data.to_csv(OUT / f"data_{case}.csv", index=False)
        truth.to_csv(OUT / f"items_{case}.csv", index=False, float_format="%.17g")
        rows.append({"case": case, "model": model, "design": design, "n": n,
                     "k": len(truth), "seed": SEED})
    pd.DataFrame(rows).to_csv(OUT / "cases.csv", index=False)
    rows = []
    for case, k, n in FREE_4PL:
        truth = free_4pl_items(k)
        data, _ = simulate_irt(n, truth, seed=SEED)
        data.to_csv(OUT / f"data_{case}.csv", index=False)
        truth.to_csv(OUT / f"items_{case}.csv", index=False, float_format="%.17g")
        rows.append({"case": case, "model": "4PL", "design": "free", "n": n, "k": k, "seed": SEED})
    pd.DataFrame(rows).to_csv(OUT / "cases_free4pl.csv", index=False)


if __name__ == "__main__":
    main()
