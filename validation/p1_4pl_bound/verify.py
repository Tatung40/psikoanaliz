"""Independent re-check of the 4PL item-1 c bound: 31-point grid, maxiter 400, KL re-evaluated with 121-node
quadrature at the same nuisance, lower Riemann (left/right min) integration, N for bound <= box/2."""
import json, sys
import numpy as np
from scipy.optimize import minimize
from mirt.scoring._common import build_quadrature
from validation.binary_certification.common import truth
from validation.p3_sx2_analytic import kernel as K

def px121(e, model):
    t, w = build_quadrature(n_quadpts=121, n_factors=1, prior_mean=None, prior_cov=None); t = np.ravel(t); w = np.ravel(w)
    P, _ = K.irf(e, model, t); ll = K.X @ np.log(P).T + (1 - K.X) @ np.log1p(-P).T
    return (np.exp(ll) * w).sum(1)

model, j, q = "4PL", 0, 2; k = 4; p = truth(model, 12); e0 = np.stack([p[s] for s in "abcd"], 1).ravel(); idx = j * k + q
p0, _ = K.pattern_terms(e0, model); q0 = px121(e0, model); free = np.array([i for i in range(48) if i != idx])
bnds = [((.05, 6) if i % 4 == 0 else (-6, 6) if i % 4 == 1 else (1e-6, .5) if i % 4 == 2 else (.5, 1 - 1e-6)) for i in free]
grid = np.unique(np.r_[np.linspace(1e-6, .5, 31), e0[idx]]); out = []; start = e0[free].copy()
for t in sorted(grid, key=lambda v: abs(v - e0[idx])):
    def f(z):
        e = e0.copy(); e[free] = z; e[idx] = t; px, sc = K.pattern_terms(e, model)
        return float(p0 @ (np.log(p0) - np.log(px))), -(p0 @ sc)[free]
    r = minimize(f, start, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 400, "ftol": 1e-15, "gtol": 1e-10})
    e = e0.copy(); e[free] = r.x; e[idx] = t; px, _ = K.pattern_terms(e, model); qx = px121(e, model)
    out.append(dict(t=float(t), kl61=max(float(r.fun), 0), kl121=float(q0 @ (np.log(q0) - np.log(qx))), bc=float(np.sum(np.sqrt(p0 * px)))))
out.sort(key=lambda d: d["t"])
def bound(N, key="kl61"):
    t = np.array([d["t"] for d in out]); kl = np.array([max(d[key], 0) for d in out]); bc = np.array([d["bc"] for d in out])
    tv = np.minimum(np.minimum(np.sqrt(N * kl / 2), np.sqrt(np.clip(1 - bc**(2 * N), 0, 1))), 1); g = np.maximum(0, .95 - tv)
    return float(np.trapezoid(g, t)), float(np.sum(np.minimum(g[:-1], g[1:]) * np.diff(t)))
res = dict(points=out, N3000_trapezoid_and_lower_riemann=bound(3000), N3000_kl121=bound(3000, "kl121"),
           N_half_box_lower_riemann=next((int(n) for n in np.geomspace(1000, 1e7, 400) if bound(n)[1] <= .25), None),
           N_half_box_trapezoid=next((int(n) for n in np.geomspace(1000, 1e7, 400) if bound(n)[0] <= .25), None))
json.dump(res, open(sys.argv[1], "w"), indent=1); print({k: v for k, v in res.items() if k != "points"})
