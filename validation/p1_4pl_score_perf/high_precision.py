"""Independent extended-precision reference for the closure-K statistic at the worst discrepancy points.
Pattern terms in numpy longdouble (64-bit mantissa) with the original per-node formula; efficient-score solve in
mpmath at 50 digits. Compares frozen (float64 einsum) and optimised (float64 GEMM) implementations."""
import json, sys, importlib.util
import numpy as np, mpmath as mp
from scipy.special import expit
from validation.binary_certification.common import simulate
from core.irt_python_mirt_adapter import fit_python_mirt
from core import irt_4pl_score_sets as NEW
spec = importlib.util.spec_from_file_location("F", "validation/p1_4pl_score_perf/frozen_k.py"); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
mp.mp.dps = 50; LD = np.longdouble


def terms_ld(eta, t, w):
    e = np.asarray(eta, LD).reshape(12, 4); a, b, c, d = e.T; t = t.astype(LD); w = w.astype(LD); X = OLD._X.astype(LD)
    z = 1 / (1 + np.exp(-(a * (t[:, None] - b)))); P = np.clip(c + (d - c) * z, LD(1e-300), 1 - LD(1e-16)); dz = z * (1 - z)
    dP = np.stack([(d - c) * dz * (t[:, None] - b), -(d - c) * dz * a, 1 - z, z], 2)
    ll = X @ np.log(P).T + (1 - X) @ np.log1p(-P).T; m = ll.max(1, keepdims=True); L = np.exp(ll - m) * w
    px = L.sum(1); post = L / px[:, None]; px = px * np.exp(m[:, 0])
    sc = np.einsum("xq,xqj,qjk->xjk", post, (X[:, None, :] - P[None]) / (P * (1 - P))[None], dP).reshape(len(X), -1)
    return px, sc


def T_hp(eta, phat, i, N, t, w):
    px, sc = terms_ld(eta, t, w); s = phat.astype(LD) @ sc; I = (px[:, None] * sc).T @ sc
    lam = [k for k in range(48) if k != i]
    A = mp.matrix([[mp.mpf(float(I[r, c])) + mp.mpf(float(I[r, c] - LD(float(I[r, c])))) for c in lam] for r in lam])
    rhs = mp.matrix([mp.mpf(float(I[r, i])) + mp.mpf(float(I[r, i] - LD(float(I[r, i])))) for r in lam])
    b = mp.lu_solve(A, rhs); sv = [mp.mpf(float(s[k])) + mp.mpf(float(s[k] - LD(float(s[k])))) for k in range(48)]
    g = sv[i] - sum(b[n] * sv[k] for n, k in enumerate(lam)); v = mp.mpf(float(I[i, i])) - sum(b[n] * mp.mpf(float(I[k, i])) for n, k in enumerate(lam))
    cond = float(mp.norm(A, 1) * mp.norm(mp.inverse(A), 1))
    return float(N * g * g / v), float(g), float(v), cond


def path_eta(M, eta, i, psi, Hi):
    anchor = np.clip(eta, M.LO, M.HI); e = np.clip(anchor + Hi[:, i] * (psi - anchor[i]) / Hi[i, i], M.LO, M.HI); e[i] = np.clip(psi, M.LO[i], M.HI[i]); return e


def analyse(r, mode, targets):
    x = simulate("4PL", 3000, 12, [202611262101, 4, r, 0]); N = 3000
    eta = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
    if mode == "c_high": eta[2::4] = .4999
    if mode == "d_low": eta[3::4] = .5 + 1e-6
    t, w = OLD._quad(); phat = np.bincount(np.asarray(x, int) @ OLD._BITS, minlength=4096) / N
    px, sc = OLD.terms(eta, t, w); Hi = np.linalg.inv((px[:, None] * sc).T @ sc)
    if targets is None:                         # locate worst grid point (float64 frozen vs optimised)
        best = None
        for i in [4 * j + q for j in range(12) for q in (2, 3)]:
            lo, hi = OLD.BOXQ[i % 4]
            for psi in np.linspace(lo, hi, OLD.G):
                e = path_eta(OLD, eta, i, psi, Hi); a = OLD.statistic(e, phat, i, N, t, w); b = NEW.statistic(e, phat, i, N, t, w)
                rel = abs(a - b) / max(1, a)
                if best is None or rel > best[0]: best = (rel, i, float(psi))
        targets = [(best[1], best[2])]
    out = []
    for i, psi in targets:
        e = path_eta(OLD, eta, i, psi, Hi); a = OLD.statistic(e, phat, i, N, t, w); b = NEW.statistic(e, phat, i, N, t, w); h, g, v, cond = T_hp(e, phat, i, N, t, w)
        out.append(dict(r=r, mode=mode, param_index=i, item=i // 4 + 1, param="abcd"[i % 4], psi=psi, T_frozen=a, T_optimised=b, T_high_precision=h,
                        abs_err_frozen=abs(a - h), abs_err_optimised=abs(b - h), rel_frozen_vs_optimised=abs(a - b) / max(1, a),
                        crit_distance_hp=h - OLD.CRIT, decision_hp=bool(h <= OLD.CRIT), decision_frozen=bool(a <= OLD.CRIT), decision_optimised=bool(b <= OLD.CRIT),
                        efficient_score_g=g, efficient_info_v=v, cond_I_nuisance=cond))
    return out


if __name__ == "__main__":
    res = analyse(3, "c_high", None) + analyse(2, "d_low", None) + analyse(794, "fit", [(46, .24), (46, .26)])
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for d in res: print({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in d.items()})
