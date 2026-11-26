"""Production CTT reliability values on deterministic fixtures (for the independent R comparison)."""
import json, sys
import numpy as np, pandas as pd
from core.reliability import compute_reliability_indices
from core import reliability_extended as RX


def fixtures():
    rng = np.random.default_rng(20261126); th = rng.normal(size=400); lam = np.array([.5, .6, .7, .8, .55, .65])
    poly = pd.DataFrame(np.clip(np.round(th[:, None] * lam * 1.4 + rng.normal(0, 1, (400, 6)) + 2), 0, 4), columns=[f"p{i}" for i in range(6)])
    b = np.linspace(-1.2, 1.2, 10)
    binary = pd.DataFrame((rng.random((400, 10)) < 1 / (1 + np.exp(-1.3 * (th[:, None] - b)))).astype(int), columns=[f"d{i}" for i in range(10)])
    return {"poly": poly, "binary": binary}


if __name__ == "__main__":
    out = {}
    for name, x in fixtures().items():
        x.to_csv(f"{sys.argv[1]}/{name}.csv", index=False)
        rel = compute_reliability_indices(x, list(x), is_dichotomous=name == "binary")
        S = np.cov(x.to_numpy(float), rowvar=False); Rm = np.corrcoef(x.to_numpy(float), rowvar=False)
        out[name] = {"alpha": rel["cronbach_alpha"], "omega_std_ml": rel["mcdonald_omega"], "kr20": rel.get("kr20"),
                     "cov": RX.reliability_family(S), "cor": RX.reliability_family(Rm)}
    json.dump(out, open(f"{sys.argv[1]}/python.json", "w"), indent=1, default=float)
