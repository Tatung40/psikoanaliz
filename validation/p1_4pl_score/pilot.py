"""Development pilot: Neyman C(alpha) efficient-score inversion for 4PL c/d (not a certification)."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from validation.binary_certification.common import simulate, truth
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.p3_sx2_analytic import kernel as K

ITEMS = (0, 5, 11); BOXQ = {2: (0.0, .5), 3: (.5, 1.0)}; G = 26; CRIT = 3.841458820694124
LO = np.array([.05, -6, 1e-6, .5 + 1e-6] * 12); HI = np.array([6, 6, .5 - 1e-6, 1 - 1e-6] * 12)


def stat(eta, phat, i, N):
    px, sc = K.pattern_terms(eta, "4PL"); s = phat @ sc; I = (px[:, None] * sc).T @ sc
    lam = np.r_[0:i, i + 1:len(eta)]; b = np.linalg.solve(I[np.ix_(lam, lam)], I[lam, i])
    g = s[i] - b @ s[lam]; v = I[i, i] - b @ I[lam, i]
    return N * g * g / v


def one(r):
    x = simulate("4PL", 3000, 12, [202611262001, 4, r, 0]); N = len(x)
    f = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit
    eta = f.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
    px, sc = K.pattern_terms(eta, "4PL"); I = (px[:, None] * sc).T @ sc; gp, _ = K.log_prior_grad_hess(eta, "4PL")
    Hi = np.linalg.inv(I); anchor = np.clip(eta, LO, HI); phat = K.pattern_counts(x) / N  # MAP anchor (in box)
    p = truth("4PL", 12); t0 = np.stack([p[s] for s in "abcd"], 1).ravel(); out = []
    for j in ITEMS:
        for q in (2, 3):
            i = 4 * j + q; lo, hi = BOXQ[q]; grid = np.linspace(lo + 1e-6, hi - 1e-6, G)
            path = lambda psi: np.clip(anchor + Hi[:, i] * (psi - anchor[i]) / Hi[i, i], LO, HI)
            acc = [stat((lambda e: (e.__setitem__(i, psi), e)[1])(path(psi)), phat, i, N) <= CRIT for psi in grid]
            e = path(t0[i]); e[i] = t0[i]; cov = stat(e, phat, i, N) <= CRIT
            out.append(dict(item=j, q="abcd"[q], covered=bool(cov), accepted=int(sum(acc)), whole=bool(all(acc)), empty=not any(acc),
                            width=float(sum(acc) * (hi - lo) / G)))
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex: res = [o for rr in ex.map(one, range(int(sys.argv[2]))) for o in rr]
    json.dump(res, open(sys.argv[1], "w"))
    for q in "cd":
        R = [o for o in res if o["q"] == q]; inf = [o for o in R if not o["whole"] and not o["empty"]]
        print(q, len(R), "cov", np.mean([o["covered"] for o in R]).round(3), "whole", np.mean([o["whole"] for o in R]).round(3),
              "empty", np.mean([o["empty"] for o in R]).round(3), "cond cov", np.mean([o["covered"] for o in inf]).round(3) if inf else None,
              "med width inf", np.median([o["width"] for o in inf]).round(3) if inf else None)
