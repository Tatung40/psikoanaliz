"""V8: independent re-implementation of RaschPy's documented pairwise composite likelihood (unregularised).
Shares no code with raschpy: own pair counting from long data, own parameterisation, scipy optimiser.
Patterns: two observations of one person by the same rater on items i < j, or on the same item by raters r < q;
(a, b) against (a-1, b+1) is a binomial comparison with logit b[v, b] - b[u, a-1], where b[cell, h] is the step
location delta_i + tau_h + rater term of step h+1 (h = 0..K-1) and u, v are the first and second observation cells."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODELS = ("global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch")


def comparisons(d, K):
    """(u_cell, v_cell, a, b, t, u) from long data d[person, item, rater, score]; cells indexed (rater, item)."""
    items = sorted(d.item.unique()); raters = sorted(d.rater.unique()); I = len(items)
    x = d.assign(c=d.rater.map({r: k for k, r in enumerate(raters)}) * I + d.item.map({i: k for k, i in enumerate(items)}))
    pairs = []
    for kind, (key, other) in enumerate((("rater", "item"), ("item", "rater"))):
        m = x.merge(x, on=["person", key], suffixes=("1", "2"))
        pairs.append(m[m[f"{other}1"] < m[f"{other}2"]][["c1", "c2", "score1", "score2"]].assign(kind=kind))
    p = pd.concat(pairs).groupby(["kind", "c1", "c2", "score1", "score2"]).size()
    rows = []
    for (kind, c1, c2), g in p.groupby(level=[0, 1, 2]):
        n = g.droplevel([0, 1, 2]).to_dict()
        for a in range(1, K + 1):
            for b in range(0, K):
                t, u = n.get((a, b), 0), n.get((a - 1, b + 1), 0)
                if t + u: rows.append((c1, c2, a, b, t, u, kind))
    return np.array(rows, float), raters, items


def _zs(v):                                        # free vector of length n-1 -> zero-sum vector of length n
    return np.append(v, -v.sum())


def _ext(A, axis):                                 # append minus the sum along axis (zero-sum extension)
    return np.concatenate([A, -A.sum(axis, keepdims=True)], axis)


def grid(model, p, R, I, K):
    """Step-location grid b[r, i, h] (R, I, K) and the facet-element blocks (for the ridge) from free parameters p.
    Linear models: lambda | a (centred over raters) | t (centred over raters) | t2 (also zero-sum over steps) |
    e (centred over raters, zero-sum over items and steps). Stretch: lambda zero-sum, omegas with mean 1."""
    k = 0
    def take(n):
        nonlocal k; out = p[k:k + n]; k += n; return out
    delta = _zs(take(I - 1)); tau = _zs(take(K - 1))
    b = delta[None, :, None] + tau[None, None, :] + np.zeros((R, I, K)); blocks = {}
    if model == "global":
        blocks["lambda"] = _zs(take(R - 1)); b += blocks["lambda"][:, None, None]
    if model in ("items", "bivector", "matrix"):
        blocks["a"] = _ext(take((R - 1) * I).reshape(R - 1, I), 0); b += blocks["a"][:, :, None]
    if model == "thresholds":
        blocks["t"] = _ext(take((R - 1) * K).reshape(R - 1, K), 0); b += blocks["t"][:, None, :]
    if model in ("bivector", "matrix"):
        blocks["t2"] = _ext(_ext(take((R - 1) * (K - 1)).reshape(R - 1, K - 1), 0), 1); b += blocks["t2"][:, None, :]
    if model == "matrix":
        blocks["e"] = _ext(_ext(_ext(take((R - 1) * (I - 1) * (K - 1)).reshape(R - 1, I - 1, K - 1), 0), 1), 2); b += blocks["e"]
    if model in ("centrality", "pseudo_halo", "bistretch"):
        lam = _zs(take(R - 1)); b += lam[:, None, None]
        if model in ("pseudo_halo", "bistretch"):
            blocks["omega_items"] = 1 + _zs(take(R - 1)); b += (blocks["omega_items"] - 1)[:, None, None] * delta[None, :, None]
        if model in ("centrality", "bistretch"):
            blocks["omega_thresholds"] = 1 + _zs(take(R - 1)); b += (blocks["omega_thresholds"] - 1)[:, None, None] * tau[None, None, :]
    return b, blocks


def n_params(model, R, I, K):
    base = I - 1 + K - 1
    return base + {"global": R - 1, "items": (R - 1) * I, "thresholds": (R - 1) * K,
                   "bivector": (R - 1) * I + (R - 1) * (K - 1),
                   "matrix": (R - 1) * I + (R - 1) * (K - 1) + (R - 1) * (I - 1) * (K - 1),
                   "centrality": 2 * (R - 1), "pseudo_halo": 2 * (R - 1), "bistretch": 3 * (R - 1)}[model]


def settings(d, model, K, regularised):
    """Regularisation of RaschPy 1.4.0's defaults (documented in MFRM.calibrate), or none."""
    R, I = d.rater.nunique(), d.item.nunique()
    if not regularised:
        return dict(c_item=0., c_thr=0., c_facet=0., ridge=0., interaction_ridge=0., stretch_ridge=0.)
    rows = d.groupby(["rater", "person"]).ngroups
    item_c = min(1.0, 0.548 * I ** (-0.2 * np.log10(max(rows * K, 1))))
    factor = {"global": 1 / (I * K), "items": 1 / K, "thresholds": 1 / I, "bivector": (I + K) / (I * K), "matrix": 1.0,
              "centrality": 1 / I, "pseudo_halo": 1 / K, "bistretch": (I + K) / (I * K)}[model]
    return dict(c_item=item_c / (R * K), c_thr=0.03 / R, c_facet=0.1 * factor, ridge=.3, interaction_ridge=3., stretch_ridge=1.)


def thin_widths(d, K):
    """Widths whose (k,k)/(k-1,k+1) same-rater item-pair counts are < 20 or one-sided (RaschPy would add a thin-width
    ridge, not re-implemented here: the comparison is only valid when this returns [])."""
    m = d.merge(d, on=["person", "rater"], suffixes=("1", "2")); m = m[m.item1 != m.item2]
    out = []
    for k in range(1, K):
        E = int(((m.score1 == k) & (m.score2 == k)).sum()); S = int(((m.score1 == k - 1) & (m.score2 == k + 1)).sum())
        if E == 0 or S == 0 or (E + S) / 2 < 20: out.append(k)
    return out


def fit(d, model, K, regularised=False):
    C, raters, items = comparisons(d, K); R, I = len(raters), len(items); s = settings(d, model, K, regularised)
    u, v, a, bb = (C[:, j].astype(int) for j in range(4)); kind = C[:, 6].astype(int); dd = np.abs(bb - (a - 1))
    c = np.where(dd == 0, np.where(kind == 0, s["c_item"], s["c_facet"]), np.where(dd == 1, s["c_thr"], 0.))
    t, uu = C[:, 4] + c, C[:, 5] + c
    pen_w = {"lambda": s["ridge"], "a": s["ridge"], "t": s["ridge"], "t2": s["ridge"], "e": s["interaction_ridge"],
             "omega_items": s["stretch_ridge"], "omega_thresholds": s["stretch_ridge"]}
    def obj(p):
        B, blocks = grid(model, p, R, I, K); B = B.reshape(R * I, K)
        eta = B[v, bb] - B[u, a - 1]
        pen = sum(pen_w[n] * float(((x - 1) ** 2 if n.startswith("omega") else x ** 2).sum()) for n, x in blocks.items())
        return float((t * np.logaddexp(0, -eta) + uu * np.logaddexp(0, eta)).sum()) + pen
    r = minimize(obj, np.zeros(n_params(model, R, I, K)), method="BFGS", options=dict(gtol=1e-9, maxiter=20000))
    return grid(model, r.x, R, I, K)[0], raters, items, r


def raschpy_grid(m, model, raters, items, K):
    """RaschPy's step grid from its fitted parameters (delta_i + tau_h + rater cell)."""
    delta = getattr(m, f"items_{model}"); tau = np.asarray(getattr(m, f"thresholds_{model}"), float)[-K:]
    Rv = getattr(m, f"raters_{model}"); B = np.zeros((len(raters), len(items), K))
    for x, r in enumerate(raters):
        for y, i in enumerate(items):
            if model == "global": s = np.full(K, Rv[r])
            elif model in ("items", "pseudo_halo"): s = np.full(K, Rv.loc[r, i])
            elif model in ("thresholds", "centrality"): s = Rv.loc[r].to_numpy(float)
            else: s = Rv.loc[(r, i)].to_numpy(float)
            B[x, y] = delta[i] + tau + s
    return B


def compare(d, model, K, regularised=False):
    import warnings, raschpy
    w = d.pivot(index=["rater", "person"], columns="item", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = raschpy.MFRM(w, max_score=K)
        if regularised: m.calibrate(model=model)
        else: m.calibrate(model=model, constant=0.0, threshold_constant=0.0, ridge=0.0, interaction_ridge=0.0, stretch_ridge=0.0)
    mine, raters, items, opt = fit(d, model, K, regularised); theirs = raschpy_grid(m, model, raters, items, K)
    diff = (mine - mine.mean()) - (theirs - theirs.mean())
    return float(np.abs(diff).max()), bool(opt.success), opt.fun
