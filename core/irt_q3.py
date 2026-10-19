"""Yen's Q3 local-dependence analysis for the 1PL and 2PL models (B10).

Residuals d_pi = x_pi - P_i(theta_hat_p) with P_i(theta) = logistic(a_i (theta - b_i))
from the supplied item parameters (1PL: a_i = 1) and theta_hat = Warm WLE (default) or
EAP under the model's N(0, 1) latent distribution. Q3_ij is the Pearson correlation of
d_i and d_j over persons who answered both items (pairwise deletion); NaN when fewer
than `min_pairs` common responses remain or a residual variance is zero. Adjusted Q3 =
Q3 - mean off-diagonal Q3 (Christensen, Makransky & Horton, 2017); because theta_hat
is estimated from the same responses, Q3 is negatively biased under local
independence (about -1/(k-1)). No universal cut-off is used as a decision rule.

Optional parametric bootstrap: data are simulated from the fitted model (item
parameters, theta ~ N(0, 1), observed missing pattern), the model is re-estimated with
the supplied `refit` (the original estimator and settings) and persons are re-scored
with the same method in every replicate; the maximum adjusted Q3 gives a Monte Carlo
p-value (1 + #{max* >= max_obs}) / (B + 1). Failed or non-converged replicates are
counted and excluded.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp
from core.rasch import _grid
from core.rasch_q3 import _adjust
from core.irt_mg import as_matrix, fit_single, Design

MAX_BOOTSTRAP = 500


def probabilities(theta, a, b, c=None, d=None):
    """P = c + (d - c) logistic(a (theta - b)); c = 0, d = 1 when omitted (1PL/2PL)."""
    L = expit(np.asarray(a, float)[None, :] * (np.asarray(theta, float)[:, None] - np.asarray(b, float)[None, :]))
    if c is None and d is None:
        return L
    c = np.zeros(L.shape[1]) if c is None else np.asarray(c, float)
    d = np.ones(L.shape[1]) if d is None else np.asarray(d, float)
    return c[None, :] + (d - c)[None, :] * L


def wle_general(x, observed, a, b, c, d, lo=-12., hi=12., iterations=80):
    """Warm (1989) WLE for c + (d - c) logistic(a(theta - b)) (3PL/4PL).

    Root of  sum_i (x_i - P_i) P_i' / (P_i Q_i) + J / (2 I),  I = sum P'^2 / (PQ),
    J = sum P' P'' / (PQ),  P' = (d - c) a L (1 - L),  P'' = (d - c) a^2 L (1 - L)(1 - 2L).
    The weighted score need not be monotone for 3PL/4PL: the estimating equation is scanned on a
    grid over [lo, hi]; persons with more than one root (non-unique WLE, seen for 4PL) get NaN and
    are flagged in `multiple` (returned as the fourth element); otherwise bisection finds the root.
    Estimates at a bracket end are flagged in `edge`.
    """
    x = np.where(observed, x, 0.); a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))

    def parts(theta):
        L = expit(a[None, :] * (theta[:, None] - b[None, :]))
        P = np.clip(c + (d - c) * L, 1e-12, 1 - 1e-12)
        d1 = (d - c) * a * L * (1 - L); d2 = d1 * a * (1 - 2 * L)
        w = np.where(observed, 1. / (P * (1 - P)), 0.)
        return P, d1, d2, w

    def weighted_score(theta):
        P, d1, d2, w = parts(theta)
        info = (w * d1 ** 2).sum(1); j = (w * d1 * d2).sum(1)
        return (w * (x - P) * d1).sum(1) + j / (2 * np.maximum(info, 1e-300))

    grid = np.linspace(lo, hi, 1201)                       # step 0.02: separates closely spaced roots
    signs = np.sign(np.column_stack([weighted_score(np.full(len(x), g)) for g in grid]))
    changes = (np.diff(signs, axis=1) != 0).sum(1)
    multiple = changes > 1
    low = np.full(len(x), lo); high = np.full(len(x), hi)
    for _ in range(iterations):
        mid = (low + high) / 2
        positive = weighted_score(mid) > 0
        low = np.where(positive, mid, low); high = np.where(positive, high, mid)
    theta = (low + high) / 2
    P, d1, _, w = parts(theta)
    info = (w * d1 ** 2).sum(1)
    edge = (np.abs(theta - lo) < 1e-6) | (np.abs(theta - hi) < 1e-6)
    theta[~observed.any(1) | multiple] = np.nan
    se = np.where((info > 0) & np.isfinite(theta), 1 / np.sqrt(np.maximum(info, 1e-300)), np.nan)
    return theta, se, edge, multiple


def wle(x, observed, a, b, lo=-12., hi=12., iterations=80):
    """Warm (1989) WLE: root of sum a_i (x_i - P_i) + J / (2 I), J = sum a^3 P Q (1 - 2P)."""
    x = np.where(observed, x, 0.); a = np.asarray(a, float); b = np.asarray(b, float)

    def weighted_score(theta):
        p = probabilities(theta, a, b)
        w = np.where(observed, p * (1 - p), 0.)
        info = (w * a ** 2).sum(1)
        j = (w * a ** 3 * (1 - 2 * p)).sum(1)
        return (np.where(observed, a * (x - p), 0.)).sum(1) + j / (2 * np.maximum(info, 1e-300))

    low = np.full(len(x), lo); high = np.full(len(x), hi)
    for _ in range(iterations):
        mid = (low + high) / 2
        positive = weighted_score(mid) > 0
        low = np.where(positive, mid, low); high = np.where(positive, high, mid)
    theta = (low + high) / 2
    p = probabilities(theta, a, b)
    info = (np.where(observed, p * (1 - p), 0.) * a ** 2).sum(1)
    none = ~observed.any(1)
    theta[none] = np.nan
    return theta, np.where(info > 0, 1 / np.sqrt(np.maximum(info, 1e-300)), np.nan)


def eap(x, observed, a, b, points=41, grid=None, c=None, d=None):
    """EAP under N(0, 1): Gauss-Hermite nodes, or an explicit (theta, log-weight) grid."""
    if grid is None:
        theta, lw = _grid(points)
    else:
        theta, lw = grid
    x1 = np.where(observed, x, 0.); x0 = np.where(observed, 1. - x, 0.)
    if c is None and d is None:
        eta = np.asarray(a, float)[None, :] * (theta[:, None] - np.asarray(b, float)[None, :])
        lp = x1 @ log_expit(eta).T + x0 @ log_expit(-eta).T + lw
    else:
        P = np.clip(probabilities(theta, a, b, c, d), 1e-12, 1 - 1e-12)
        lp = x1 @ np.log(P).T + x0 @ np.log1p(-P).T + lw
    post = np.exp(lp - logsumexp(lp, axis=1)[:, None])
    mean = post @ theta
    return mean, np.sqrt(np.maximum(post @ theta ** 2 - mean ** 2, 0.))


def q3_matrix(x, observed, theta, a, b, min_pairs=30, c=None, d=None):
    resid = np.where(observed, x - probabilities(theta, a, b, c, d), np.nan)
    k = resid.shape[1]; q3 = np.full((k, k), np.nan); counts = np.zeros((k, k), int)
    for i in range(k):
        for j in range(i + 1, k):
            m = observed[:, i] & observed[:, j] & np.isfinite(resid[:, i]) & np.isfinite(resid[:, j])
            counts[i, j] = counts[j, i] = int(m.sum())
            if m.sum() < min_pairs:
                continue
            u, v = resid[m, i], resid[m, j]
            su, sv = u.std(), v.std()
            if su <= 1e-12 or sv <= 1e-12:
                continue
            q3[i, j] = q3[j, i] = float(np.mean((u - u.mean()) * (v - v.mean())) / (su * sv))
    return q3, counts


def _persons(x, obs, a, b, method, points, c=None, d=None):
    if method == "WLE":
        return wle(x, obs, a, b)[0] if c is None and d is None else wle_general(x, obs, a, b, c, d)[0]   # NaN: non-unique
    if method == "EAP":
        return eap(x, obs, a, b, points, c=c, d=d)[0]
    raise ValueError("Kişi kestirimi WLE veya EAP olmalıdır.")


def default_refit(model, points=41):
    if model in ("3PL", "4PL"):
        from core.irt_mg34 import fit_single as fit34, Design34

        def refit34(frame):
            f = fit34(frame, model, points=points, information=False)
            aa, bb, cc, dd = Design34(model, ["anchor"] * frame.shape[1], 1).unpack(f["params"], 0)
            return aa, bb, cc, dd, f["converged"]
        return refit34

    def refit(frame):
        f = fit_single(frame, model, points=points, information=False)
        a, b = Design(model, ["anchor"] * frame.shape[1], 1).unpack(f["params"], 0)
        return a, b, f["converged"]
    return refit


def q3_analysis(data, a, b, model, person_method="WLE", min_pairs=30, bootstrap=0, seed=2026,
                points=41, refit=None, converged=True, c=None, d=None):
    """Q3 for fixed 1PL/2PL/3PL/4PL item parameters; optional parametric bootstrap.

    3PL/4PL: c (and d for 4PL) are required and the full IRF is used for residuals,
    persons and simulation; `refit` must return (a, b, c, d, converged) for these models.
    """
    if model not in ("1PL", "2PL", "3PL", "4PL"):
        raise ValueError("Geçersiz model.")
    full = model in ("3PL", "4PL")
    if full:
        if c is None:
            raise ValueError("3PL/4PL Q3 için c parametreleri gereklidir.")
        c = np.asarray(c, float); d = np.ones(len(c)) if model == "3PL" or d is None else np.asarray(d, float)
        if model == "4PL" and not np.all((0 <= c) & (c < d) & (d <= 1)):
            raise ValueError("4PL için 0 ≤ c < d ≤ 1 olmalıdır.")
        if model == "3PL" and not np.all((0 <= c) & (c < 1)):
            raise ValueError("3PL için 0 ≤ c < 1 olmalıdır.")
    else:
        c = d = None
    if person_method not in ("WLE", "EAP"):
        raise ValueError("Kişi kestirimi WLE veya EAP olmalıdır.")
    if not (isinstance(bootstrap, (int, np.integer)) and 0 <= bootstrap <= MAX_BOOTSTRAP):
        raise ValueError(f"Bootstrap tekrar sayısı 0–{MAX_BOOTSTRAP} olmalıdır.")
    x, obs = as_matrix(data)
    a = np.asarray(a, float); b = np.asarray(b, float)
    if model == "1PL" and not np.allclose(a, 1.):
        raise ValueError("1PL için tüm ayırt edicilikler 1 olmalıdır.")
    k = x.shape[1]
    if len(a) != k or len(b) != k:
        raise ValueError("Madde parametresi sayısı veriyle uyuşmuyor.")
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(k)]
    theta = _persons(x, obs, a, b, person_method, points, c, d)
    q3, counts = q3_matrix(x, obs, theta, a, b, min_pairs, c, d)
    adj, mean = _adjust(q3)
    iu = np.triu_indices(k, 1)
    pairs = pd.DataFrame({"Madde 1": [names[i] for i in iu[0]], "Madde 2": [names[j] for j in iu[1]],
                          "Q3": q3[iu], "Düzeltilmiş Q3": adj[iu], "Ortak yanıt": counts[iu]})
    pairs = pairs.sort_values("Düzeltilmiş Q3", ascending=False, na_position="last").reset_index(drop=True)
    out = {"q3": pd.DataFrame(q3, index=names, columns=names), "adjusted": pd.DataFrame(adj, index=names, columns=names),
           "pairs": pairs, "mean_q3": mean, "expected_under_li": -1. / (k - 1), "person_method": person_method,
           "min_pairs": int(min_pairs), "n_undefined": int(pairs["Q3"].isna().sum()), "model": model,
           "theta": theta, "bootstrap": None, "fit_converged": bool(converged),
           "n_person_undefined": int((~np.isfinite(theta) & obs.any(1)).sum())}
    if bootstrap:
        refit = refit or default_refit(model, points)
        rng = np.random.default_rng(seed)
        observed_max = float(np.nanmax(adj[iu]))
        maxima, vals, failed = [], [], 0
        for _ in range(int(bootstrap)):
            th = rng.normal(0., 1., len(x))
            sim = (rng.random(x.shape) < probabilities(th, a, b, c, d)).astype(float)
            sim[~obs] = np.nan
            try:
                got = refit(pd.DataFrame(sim, columns=names))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failed += 1; continue
            if full:
                aa, bb, cc, dd, ok = got
            else:
                (aa, bb, ok), cc, dd = got, None, None
            if not ok:
                failed += 1; continue
            xs, os_ = as_matrix(sim)
            t = _persons(xs, os_, aa, bb, person_method, points, cc, dd)
            qs, _ = q3_matrix(xs, os_, t, aa, bb, min_pairs, cc, dd)
            aq, _ = _adjust(qs)
            maxima.append(float(np.nanmax(aq[iu]))); vals.append(aq[iu])
        maxima = np.array(maxima); used = len(maxima)
        upper = np.nanquantile(np.array(vals), .95, axis=0) if used else np.full(len(iu[0]), np.nan)
        ref = pd.DataFrame({"Madde 1": [names[i] for i in iu[0]], "Madde 2": [names[j] for j in iu[1]],
                            "Bootstrap %95 (çift)": upper})
        out["pairs"] = out["pairs"].merge(ref, on=["Madde 1", "Madde 2"], how="left")
        out["bootstrap"] = {"replications": int(bootstrap), "used": used, "failed": failed, "seed": int(seed),
                            "observed_max_adjusted_q3": observed_max,
                            "p_max": (1 + int((maxima >= observed_max).sum())) / (used + 1) if used else np.nan,
                            "max_quantile_95": float(np.quantile(maxima, .95)) if used else np.nan,
                            "max_quantile_99": float(np.quantile(maxima, .99)) if used else np.nan}
    return out
