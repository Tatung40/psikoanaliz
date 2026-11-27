"""EXPLORATORY (not confirmatory): observed-score DIF null rejection under impact, Rasch vs 2PL generators, purify on/off.
R=100 datasets per cell, seed root 20261127041; dataset-clustered MCSE. Purpose: diagnose mechanism only."""
import json, sys, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from core.dif_classical import dif_screen
warnings.filterwarnings("ignore")
K = 12; B = np.linspace(-1.5, 1.5, K)

def one(job):
    gen, mu, r = job; rng = np.random.default_rng([20261127041, ["rasch", "2pl"].index(gen), int(mu * -10), r])
    a = np.ones(K) if gen == "rasch" else np.linspace(.6, 1.8, K)
    th = np.r_[rng.normal(0, 1, 500), rng.normal(mu, 1, 500)]; g = np.array(["R"] * 500 + ["F"] * 500)
    x = (rng.random((1000, K)) < 1 / (1 + np.exp(-a * (th[:, None] - B)))).astype(int)
    out = {}
    for m in ("MH", "LR"):
        for pur in (False, True):
            t = dif_screen(x, g, "R", "F", method=m, purify=pur, adjust="none")[0]
            out[f"{m}{'_pur' if pur else ''}"] = float((t["p"] < .05).mean())
    return gen, mu, out

if __name__ == "__main__":
    jobs = [(gen, mu, r) for gen in ("rasch", "2pl") for mu in (0.0, -0.5) for r in range(100)]
    with ProcessPoolExecutor(2) as ex: res = list(ex.map(one, jobs, chunksize=10))
    rep = {}
    for gen in ("rasch", "2pl"):
        for mu in (0.0, -0.5):
            R = [o for g_, m_, o in res if g_ == gen and m_ == mu]
            rep[f"{gen} muF={mu}"] = {k: dict(type1=float(np.mean([o[k] for o in R])), cluster_mcse=float(np.std([o[k] for o in R], ddof=1) / np.sqrt(len(R)))) for k in R[0]}
    json.dump(rep, open(sys.argv[1], "w"), indent=1); print(json.dumps({k: {m: round(v["type1"], 3) for m, v in d.items()} for k, d in rep.items()}))
