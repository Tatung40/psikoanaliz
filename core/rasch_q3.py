"""Yen's (1984) Q3 local-dependence analysis for the dichotomous Rasch model (B9).

Residuals d_ni = x_ni - P(theta_hat_n, b_i) with theta_hat = Warm WLE (default) or EAP,
computed from the B8 MML item difficulties. Q3_ij is the Pearson correlation of d_i and
d_j over persons who answered both items (pairwise deletion); it is NaN when fewer than
`min_pairs` persons remain or a residual variance is zero.

Q3 is negatively biased because theta_hat is estimated from the same responses
(expected value about -1/(k-1) under local independence). Adjusted Q3 subtracts the
mean of all off-diagonal Q3 values (centering convention of Christensen, Makransky &
Horton, 2017). No universal cutoff is used as a test: pairs are ranked, and an
optional parametric bootstrap gives a Monte Carlo reference for the largest adjusted Q3
(data simulated from the fitted model with the observed missing pattern; items
re-calibrated and persons re-scored in every replicate).
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from core.rasch import fit_rasch, wle, eap


def q3_matrix(x, observed, theta, b, min_pairs=30):
    x = np.asarray(x, float); observed = np.asarray(observed, bool)
    resid = np.where(observed, np.nan_to_num(x) - expit(np.asarray(theta)[:, None] - np.asarray(b)[None, :]), np.nan)
    k = resid.shape[1]; q3 = np.full((k, k), np.nan); counts = np.zeros((k, k), int)
    for i in range(k):
        for j in range(i + 1, k):
            m = observed[:, i] & observed[:, j]
            counts[i, j] = counts[j, i] = int(m.sum())
            if m.sum() < min_pairs:
                continue
            a, c = resid[m, i], resid[m, j]
            sa, sc = a.std(), c.std()
            if sa <= 1e-12 or sc <= 1e-12:
                continue
            q3[i, j] = q3[j, i] = float(np.mean((a - a.mean()) * (c - c.mean())) / (sa * sc))
    return q3, counts, resid


def _adjust(q3):
    off = q3[np.triu_indices_from(q3, 1)]
    mean = float(np.nanmean(off)) if np.isfinite(off).any() else np.nan
    return q3 - mean, mean


def q3_analysis(data, b=None, sd=None, person_method="WLE", min_pairs=30, bootstrap=0, seed=2026,
                points=41, max_bootstrap=500):
    """Q3, adjusted Q3, ranked pairs and an optional parametric-bootstrap reference."""
    if person_method not in ("WLE", "EAP"):
        raise ValueError("Kişi kestirimi WLE veya EAP olmalıdır.")
    if not (isinstance(bootstrap, (int, np.integer)) and 0 <= bootstrap <= max_bootstrap):
        raise ValueError(f"Bootstrap tekrar sayısı 0–{max_bootstrap} olmalıdır.")
    fit = fit_rasch(data, points=points)
    x, obs = fit["data"], fit["observed"]
    b = fit["b"] if b is None else np.asarray(b, float)
    sd = fit["sd"] if sd is None else float(sd)
    def persons(xx, oo, bb, ss):
        return wle(xx, oo, bb)[0] if person_method == "WLE" else eap(xx, oo, bb, ss, points)[0]
    theta = persons(x, obs, b, sd)
    q3, counts, _ = q3_matrix(x, obs, theta, b, min_pairs)
    adj, mean = _adjust(q3)
    names = list(fit["items"]["Madde"]); k = len(names)
    rows = [{"Madde 1": names[i], "Madde 2": names[j], "Q3": q3[i, j], "Düzeltilmiş Q3": adj[i, j],
             "Ortak yanıt": counts[i, j]} for i in range(k) for j in range(i + 1, k)]
    pairs = pd.DataFrame(rows).sort_values("Düzeltilmiş Q3", ascending=False, na_position="last").reset_index(drop=True)
    result = {"q3": pd.DataFrame(q3, index=names, columns=names),
              "adjusted": pd.DataFrame(adj, index=names, columns=names), "pairs": pairs,
              "mean_q3": mean, "expected_under_li": -1. / (k - 1), "person_method": person_method,
              "min_pairs": int(min_pairs), "n_undefined": int(pairs["Q3"].isna().sum()),
              "bootstrap": None, "fit_converged": fit["converged"]}
    if bootstrap:
        rng = np.random.default_rng(seed)
        observed_max = float(np.nanmax(adj[np.triu_indices(k, 1)]))
        maxima, pair_vals, failed = [], [], 0
        for _ in range(int(bootstrap)):
            th = rng.normal(0., sd, len(x))
            sim = (rng.random(x.shape) < expit(th[:, None] - b[None, :])).astype(float)
            sim[~obs] = np.nan
            try:
                f = fit_rasch(pd.DataFrame(sim), points=points)
            except ValueError:
                failed += 1; continue
            if not f["converged"]:
                failed += 1; continue
            t = persons(f["data"], f["observed"], f["b"], f["sd"])
            qs, _, _ = q3_matrix(f["data"], f["observed"], t, f["b"], min_pairs)
            a, _ = _adjust(qs)
            vals = a[np.triu_indices(k, 1)]
            maxima.append(float(np.nanmax(vals))); pair_vals.append(vals)
        maxima = np.array(maxima)
        used = len(maxima)
        p = (1 + int((maxima >= observed_max).sum())) / (used + 1) if used else np.nan
        upper = np.nanquantile(np.array(pair_vals), .95, axis=0) if used else np.full(k * (k - 1) // 2, np.nan)
        ref = pd.DataFrame({"Madde 1": [names[i] for i, j in zip(*np.triu_indices(k, 1))],
                            "Madde 2": [names[j] for i, j in zip(*np.triu_indices(k, 1))],
                            "Bootstrap %95 (çift)": upper})
        result["pairs"] = result["pairs"].merge(ref, on=["Madde 1", "Madde 2"], how="left")
        result["bootstrap"] = {"replications": int(bootstrap), "used": used, "failed": failed, "seed": int(seed),
                               "observed_max_adjusted_q3": observed_max, "p_max": p,
                               "max_quantile_95": float(np.quantile(maxima, .95)) if used else np.nan,
                               "max_quantile_99": float(np.quantile(maxima, .99)) if used else np.nan}
    return result
