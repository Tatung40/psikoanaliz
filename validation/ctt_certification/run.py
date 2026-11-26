"""CTT-C1 confirmation runner (PROTOCOL.json). Vectorised data generation, 2 workers, JSONL checkpoints."""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import multivariate_normal, norm
from core.reliability import compute_reliability_indices
from core import reliability_extended as RX
warnings.filterwarnings("ignore")


def cells():
    out = []
    for model in ("tau", "congeneric"):
        for N in (200, 1000):
            for k in (6, 20): out.append(dict(kind="continuous", model=model, N=N, k=k))
    for N in (200, 1000):
        for k in (6, 20): out.append(dict(kind="binary", model="probit", N=N, k=k))
    return out


def population(c):
    k = c["k"]
    if c["kind"] == "continuous":
        lam = np.full(k, .6) if c["model"] == "tau" else np.linspace(.4, .8, k)
        sd = np.ones(k) if c["model"] == "tau" else np.linspace(.5, 2, k)
        L = lam * sd; psi = (1 - lam ** 2) * sd ** 2; S = np.outer(L, L) + np.diag(psi)
        est = dict(alpha=None, omega_std=lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum()), omega_raw=L.sum() ** 2 / (L.sum() ** 2 + psi.sum()))
        est["true_reliability"] = est["omega_raw"]
        return S, est, (L, psi)
    lam = np.linspace(.5, .8, k); tau = np.linspace(-1, 1, k); p = 1 - norm.cdf(tau); S = np.diag(p * (1 - p))
    for i in range(k):
        for j in range(i + 1, k):
            r = lam[i] * lam[j]
            p11 = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-tau[i], -tau[j]])
            S[i, j] = S[j, i] = p11 - p[i] * p[j]
    return S, {}, (lam, tau)


def estimands(c):
    S, est, gen = population(c); k = c["k"]
    a = k / (k - 1) * (1 - np.trace(S) / S.sum())
    est.update(alpha=a, feldt_brennan=RX.feldt_brennan(S), feldt_gilmer=RX.feldt_gilmer(S))
    if c["kind"] == "binary": est["kr20"] = a
    return S, est, gen


ROOT = int(os.environ.get("CTT_ROOT", "20261126015"))


def one(job):
    ci, r = job; c = cells()[ci]; S, est, gen = estimands(c); k, N = c["k"], c["N"]
    rng = np.random.default_rng([ROOT, ci, r])
    if c["kind"] == "continuous":
        L, psi = gen; x = rng.standard_normal((N, 1)) * L + rng.standard_normal((N, k)) * np.sqrt(psi)
    else:
        lam, tau = gen; x = ((rng.standard_normal((N, 1)) * lam + rng.standard_normal((N, k)) * np.sqrt(1 - lam ** 2)) > tau).astype(float)
    df = pd.DataFrame(x, columns=[f"i{j}" for j in range(k)])
    rec = dict(cell=ci, r=r)
    try:
        rel = compute_reliability_indices(df, list(df), is_dichotomous=c["kind"] == "binary")
        fam = RX.reliability_family(np.cov(x, rowvar=False, ddof=1))
        rec.update(alpha=rel["cronbach_alpha"], feldt_brennan=fam["feldt_brennan"], feldt_gilmer=fam["feldt_gilmer"])
        if c["kind"] == "binary": rec["kr20"] = rel.get("kr20")
        else: rec.update(omega_std=rel["mcdonald_omega"], omega_raw=rel["mcdonald_omega_raw"])
    except Exception as e:
        rec["error"] = repr(e)
    return rec


if __name__ == "__main__":
    out = Path(sys.argv[1]); R = int(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
    done = {(d["cell"], d["r"]) for d in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    only = [int(v) for v in os.environ.get("CTT_CELLS", "").split(",") if v] or range(len(cells()))
    jobs = [(ci, r) for ci in only for r in range(R) if (ci, r) not in done]
    with ProcessPoolExecutor(os.cpu_count()) as ex, out.open("a") as fh:
        for rec in ex.map(one, jobs, chunksize=20): fh.write(json.dumps(rec) + "\n")
