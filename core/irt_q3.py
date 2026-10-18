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


def probabilities(theta, a, b):
    return expit(np.asarray(a, float)[None, :] * (np.asarray(theta, float)[:, None] - np.asarray(b, float)[None, :]))


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


def eap(x, observed, a, b, points=41, grid=None):
    """EAP under N(0, 1): Gauss-Hermite nodes, or an explicit (theta, log-weight) grid."""
    if grid is None:
        theta, lw = _grid(points)
    else:
        theta, lw = grid
    x1 = np.where(observed, x, 0.); x0 = np.where(observed, 1. - x, 0.)
    eta = np.asarray(a, float)[None, :] * (theta[:, None] - np.asarray(b, float)[None, :])
    lp = x1 @ log_expit(eta).T + x0 @ log_expit(-eta).T + lw
    post = np.exp(lp - logsumexp(lp, axis=1)[:, None])
    mean = post @ theta
    return mean, np.sqrt(np.maximum(post @ theta ** 2 - mean ** 2, 0.))


def q3_matrix(x, observed, theta, a, b, min_pairs=30):
    resid = np.where(observed, x - probabilities(theta, a, b), np.nan)
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


def _persons(x, obs, a, b, method, points):
    if method == "WLE":
        return wle(x, obs, a, b)[0]
    if method == "EAP":
        return eap(x, obs, a, b, points)[0]
    raise ValueError("Kişi kestirimi WLE veya EAP olmalıdır.")


def default_refit(model, points=41):
    def refit(frame):
        f = fit_single(frame, model, points=points, information=False)
        a, b = Design(model, ["anchor"] * frame.shape[1], 1).unpack(f["params"], 0)
        return a, b, f["converged"]
    return refit


def q3_analysis(data, a, b, model, person_method="WLE", min_pairs=30, bootstrap=0, seed=2026,
                points=41, refit=None, converged=True):
    """Q3 for fixed 1PL/2PL item parameters; optional parametric bootstrap."""
    if model not in ("1PL", "2PL"):
        raise ValueError("Bu Q3 uygulaması yalnızca 1PL ve 2PL içindir.")
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
    theta = _persons(x, obs, a, b, person_method, points)
    q3, counts = q3_matrix(x, obs, theta, a, b, min_pairs)
    adj, mean = _adjust(q3)
    iu = np.triu_indices(k, 1)
    pairs = pd.DataFrame({"Madde 1": [names[i] for i in iu[0]], "Madde 2": [names[j] for j in iu[1]],
                          "Q3": q3[iu], "Düzeltilmiş Q3": adj[iu], "Ortak yanıt": counts[iu]})
    pairs = pairs.sort_values("Düzeltilmiş Q3", ascending=False, na_position="last").reset_index(drop=True)
    out = {"q3": pd.DataFrame(q3, index=names, columns=names), "adjusted": pd.DataFrame(adj, index=names, columns=names),
           "pairs": pairs, "mean_q3": mean, "expected_under_li": -1. / (k - 1), "person_method": person_method,
           "min_pairs": int(min_pairs), "n_undefined": int(pairs["Q3"].isna().sum()), "model": model,
           "theta": theta, "bootstrap": None, "fit_converged": bool(converged)}
    if bootstrap:
        refit = refit or default_refit(model, points)
        rng = np.random.default_rng(seed)
        observed_max = float(np.nanmax(adj[iu]))
        maxima, vals, failed = [], [], 0
        for _ in range(int(bootstrap)):
            th = rng.normal(0., 1., len(x))
            sim = (rng.random(x.shape) < probabilities(th, a, b)).astype(float)
            sim[~obs] = np.nan
            try:
                aa, bb, ok = refit(pd.DataFrame(sim, columns=names))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failed += 1; continue
            if not ok:
                failed += 1; continue
            xs, os_ = as_matrix(sim)
            t = _persons(xs, os_, aa, bb, person_method, points)
            qs, _ = q3_matrix(xs, os_, t, aa, bb, min_pairs)
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
