"""Fresh-seed confirmation of closure K (4PL c/d C(alpha) score sets). JSONL, resumable."""
import argparse, json, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from validation.binary_certification.common import simulate, truth
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_4pl_score_sets import confidence_sets


def one(job):
    root, r = job; rec = {"r": r, "seed": [root, 4, r, 0]}
    try:
        x = simulate("4PL", 3000, 12, rec["seed"])
        f = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit
        rec["converged"] = bool(f.converged)
        if not f.converged: return rec
        eta = f.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
        p = truth("4PL", 12); t0 = np.stack([p[s] for s in "abcd"], 1).ravel()
        rec["sets"] = confidence_sets(eta, x, truth=t0)
    except Exception as e:
        rec["error"] = repr(e)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=int); ap.add_argument("--R", type=int); ap.add_argument("--start", type=int); ap.add_argument("--step", type=int); ap.add_argument("--out")
    a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(os.cpu_count()) as ex, out.open("a") as fh:
        for rec in ex.map(one, [(a.root, r) for r in range(a.start, a.R, a.step)]): fh.write(json.dumps(rec) + "\n"); fh.flush()
