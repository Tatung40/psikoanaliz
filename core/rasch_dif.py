"""Two-group uniform DIF for the dichotomous Rasch model (B9).

Model (concurrent multigroup MML): P(X=1 | theta, g) = logistic(theta - b_ig),
theta ~ N(mu_g, sigma_g^2). Identification extends the B8 convention: reference
group mean mu_R = 0; mu_F, sigma_R, sigma_F free; anchor items share one difficulty in
both groups, studied items get b_iR and b_iF. Group ability differences are therefore
absorbed by mu_F/sigma_F and are not mistaken for DIF.

Estimand and sign: Delta_i = b_iF - b_iR (logits) for a studied item given the anchor
set; Delta > 0 means the item is harder for the focal group at equal theta.
Wald z = Delta / SE, SE from the joint observed information (anchor, latent-distribution
and item uncertainty included). The anchor set is chosen from the data (all-other-items
start, iterative purification), so p-values are conditional on the selected anchors;
this selection effect is not part of the SE (documented limitation). A significant
result is evidence of differential functioning, not proof of item bias.

quadrature="gauss-hermite" (production, theta = mu + sigma z on GH nodes) or "mirt"
(61 equally spaced nodes on [-6, 6], weights proportional to the normal density; used
only to reproduce R mirt multipleGroup exactly in validation).
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import norm
from core.rasch import _as_matrix, _grid, SCORE_TOLERANCE

B_BOUND = 10.
MU_BOUND = 6.
LOG_SD_BOUNDS = (np.log(1e-2), np.log(10.))
ETS_B, ETS_C = 1.0 / 2.35, 1.5 / 2.35          # ETS delta 1.0 / 1.5 in logits (D = 2.35 logit)


def _design(k, studied):
    """Index of each item's difficulty in the parameter vector, per group."""
    studied = sorted(set(studied))
    idx_r, idx_f, labels, pos = np.zeros(k, int), np.zeros(k, int), [], 0
    for i in range(k):
        if i in studied:
            idx_r[i], idx_f[i] = pos, pos + 1; labels += [("R", i), ("F", i)]; pos += 2
        else:
            idx_r[i] = idx_f[i] = pos; labels.append(("A", i)); pos += 1
    return idx_r, idx_f, labels, pos                 # pos = first latent parameter index


class _Groups:
    def __init__(self, x, obs, groups, quadrature, points):
        self.parts = []
        for g in (0, 1):
            m = groups == g
            self.parts.append((np.where(obs[m], x[m], 0.), np.where(obs[m], 1. - x[m], 0.)))
        self.quadrature = quadrature
        if quadrature == "gauss-hermite":
            self.z, self.lw = _grid(points)
        elif quadrature == "mirt":
            self.theta = np.linspace(-6., 6., 61)
        else:
            raise ValueError("Bilinmeyen kuadratür.")

    def nodes(self, mu, sd):
        if self.quadrature == "gauss-hermite":
            return mu + sd * self.z, self.lw
        lw = norm.logpdf(self.theta, mu, sd)
        return self.theta, lw - logsumexp(lw)


def _negloglik(params, G, idx_r, idx_f, latent, n_total):
    mu = (0., params[latent]); sd = (np.exp(params[latent + 1]), np.exp(params[latent + 2]))
    grad = np.zeros_like(params); value = 0.
    for g, idx in ((0, idx_r), (1, idx_f)):
        x1, x0 = G.parts[g]
        theta, lw = G.nodes(mu[g], sd[g])
        b = params[idx]
        eta = theta[:, None] - b[None, :]
        lp = x1 @ log_expit(eta).T + x0 @ log_expit(-eta).T + lw
        normc = logsumexp(lp, axis=1); value -= normc.sum()
        post = np.exp(lp - normc[:, None])
        p = expit(eta)
        resid = post.T @ x1 - (post.T @ (x1 + x0)) * p           # Q x k, d/d eta
        np.add.at(grad, idx, resid.sum(axis=0))                   # d/d b = -sum_q resid
        d_theta = resid.sum(axis=1)
        if G.quadrature == "gauss-hermite":
            g_mu, g_ls = d_theta.sum(), (d_theta * sd[g] * G.z).sum()
        else:                                                     # weights depend on (mu, sd)
            r_q = post.sum(axis=0); w = np.exp(lw); n_g = r_q.sum()
            u = (theta - mu[g]) / sd[g] ** 2; v = (theta - mu[g]) ** 2 / sd[g] ** 2
            g_mu = (r_q * u).sum() - n_g * (w * u).sum()
            g_ls = (r_q * v).sum() - n_g * (w * v).sum()
        if g == 1:
            grad[latent] -= g_mu
        grad[latent + 1 + g] -= g_ls
    return value / n_total, grad / n_total


def fit_multigroup(x, obs, groups, studied, quadrature="gauss-hermite", points=41, max_iter=3000):
    k = x.shape[1]
    idx_r, idx_f, labels, latent = _design(k, studied)
    G = _Groups(x, obs, groups, quadrature, points)
    p = np.clip(np.nansum(np.where(obs, x, 0.), 0) / obs.sum(0), .02, .98)
    b0 = -np.log(p / (1 - p))
    start = np.zeros(latent + 3)
    for j, (_, i) in enumerate(labels):
        start[j] = b0[i]
    bounds = [(-B_BOUND, B_BOUND)] * latent + [(-MU_BOUND, MU_BOUND), LOG_SD_BOUNDS, LOG_SD_BOUNDS]
    n = len(x)
    res = minimize(_negloglik, start, args=(G, idx_r, idx_f, latent, n), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12})
    params = np.asarray(res.x, float)
    score = _negloglik(params, G, idx_r, idx_f, latent, n)[1]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    at_bound = (np.abs(params - lo) < 1e-6) | (np.abs(params - hi) < 1e-6)
    max_score = float(np.abs(score[~at_bound]).max()) if (~at_bound).any() else 0.
    converged = bool(np.isfinite(params).all() and max_score <= SCORE_TOLERANCE and not at_bound.any())
    cov, reason = None, ""
    if not converged:
        reason = "Çok gruplu kestirim yakınsamadı veya bir parametre sınırda."
    else:
        m = len(params); info = np.zeros((m, m))
        for j in range(m):
            h = 1e-5 * max(1., abs(params[j]))
            up, dn = params.copy(), params.copy(); up[j] += h; dn[j] -= h
            info[:, j] = (_negloglik(up, G, idx_r, idx_f, latent, n)[1]
                          - _negloglik(dn, G, idx_r, idx_f, latent, n)[1]) * n / (2 * h)
        info = (info + info.T) / 2
        eig = np.linalg.eigvalsh(info)
        if not np.isfinite(eig).all() or eig[0] <= 1e-8 * max(eig[-1], 1.):
            reason = "Gözlenen bilgi matrisi pozitif tanımlı değil; SH üretilemez."
        else:
            cov = np.linalg.inv(info)
    return {"params": params, "cov": cov, "se_reason": reason, "converged": converged,
            "loglik": -_negloglik(params, G, idx_r, idx_f, latent, n)[0] * n,
            "max_score_per_person": max_score, "idx_r": idx_r, "idx_f": idx_f, "latent": latent,
            "b_R": params[idx_r], "b_F": params[idx_f], "mu_F": float(params[latent]),
            "sd_R": float(np.exp(params[latent + 1])), "sd_F": float(np.exp(params[latent + 2])),
            "studied": sorted(set(studied)), "quadrature": quadrature, "iterations": int(res.nit)}


def item_contrast(fit, i):
    """Delta_i = b_iF - b_iR with SE from the joint covariance (NaN if unavailable)."""
    a, b = fit["idx_r"][i], fit["idx_f"][i]
    delta = float(fit["params"][b] - fit["params"][a])
    if fit["cov"] is None or a == b:
        return delta, np.nan
    c = fit["cov"]
    return delta, float(np.sqrt(c[a, a] + c[b, b] - 2 * c[a, b]))


def benjamini_hochberg(p):
    p = np.asarray(p, float); out = np.full(p.shape, np.nan)
    ok = np.isfinite(p); m = int(ok.sum())
    if m:
        order = np.argsort(p[ok]); ranked = p[ok][order] * m / np.arange(1, m + 1)
        q = np.minimum.accumulate(ranked[::-1])[::-1]
        tmp = np.empty(m); tmp[order] = np.minimum(q, 1.)
        out[ok] = tmp
    return out


def ets_class(delta, se, significant):
    """ETS-like A/B/C on the logit scale (Zwick, 2012; D = 2.35 logit). Approximate."""
    if not np.isfinite(delta) or not significant:
        return "A (ihmal edilebilir)"
    if abs(delta) >= ETS_C and np.isfinite(se) and abs(delta) - ETS_B > 1.96 * se:
        return "C (büyük)"
    return "B (orta)" if abs(delta) >= ETS_B else "A (ihmal edilebilir)"


def _prepare(data, group, reference, focal):
    x, obs = _as_matrix(data)
    g = pd.Series(group).reset_index(drop=True)
    if len(g) != len(x):
        raise ValueError("Grup değişkeni ile veri satır sayısı uyuşmuyor.")
    keep = g.isin([reference, focal]).to_numpy() & obs.any(axis=1)
    x, obs = x[keep], obs[keep]
    groups = (g[keep] == focal).to_numpy().astype(int)
    return x, obs, groups, keep


def dif_analysis(data, group, reference, focal, alpha=.05, purify=True, max_rounds=10,
                 min_anchors=None, quadrature="gauss-hermite", points=41, min_group=100):
    """Uniform DIF: all-other-items start, iterative anchor purification, Wald, BH-FDR."""
    if reference == focal:
        raise ValueError("Referans ve odak grup farklı olmalıdır.")
    if not 0 < alpha < .5:
        raise ValueError("Anlamlılık düzeyi 0 ile 0,5 arasında olmalıdır.")
    x, obs, groups, keep = _prepare(data, group, reference, focal)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(x.shape[1])]
    k = x.shape[1]
    sizes = {reference: int((groups == 0).sum()), focal: int((groups == 1).sum())}
    warnings = []
    if min(sizes.values()) < 30:
        raise ValueError(f"Her grupta en az 30 kişi gereklidir (bulunan: {sizes}).")
    if min(sizes.values()) < min_group:
        warnings.append(f"Grup büyüklükleri küçük ({sizes}); DIF kestirimleri kararsız ve testlerin gücü "
                        "düşük olabilir.")
    # Items testable for DIF need both responses observed in each group.
    testable, untestable = [], {}
    for i in range(k):
        reasons = []
        for g, label in ((0, reference), (1, focal)):
            o = obs[groups == g, i]; v = x[groups == g, i][o]
            if o.sum() == 0:
                reasons.append(f"{label} grubunda hiç yanıt yok")
            elif v.min() == v.max():
                reasons.append(f"{label} grubunda tüm yanıtlar aynı")
        if reasons:
            untestable[i] = "; ".join(reasons)
        else:
            testable.append(i)
    linking = [i for i in range(k) if obs[groups == 0, i].any() and obs[groups == 1, i].any()]
    if len(linking) < 2:
        raise ValueError("Bağlantısız tasarım: iki grubun ortak yanıtladığı madde sayısı yetersiz.")
    if untestable:
        warnings.append("DIF için test edilemeyen maddeler (yalnızca çapa olabilir): "
                        + "; ".join(f"{names[i]}: {r}" for i, r in untestable.items()))
    min_anchors = max(2, math.ceil(.2 * k)) if min_anchors is None else int(min_anchors)

    def test_items(anchor_pool):
        rows = {}
        for i in testable:
            anchors = [j for j in anchor_pool if j != i and j in linking]
            if len(anchors) < 1:
                rows[i] = (np.nan, np.nan, "Çapa yok"); continue
            fit = fit_multigroup(x, obs, groups, [j for j in range(k) if j not in anchors],
                                 quadrature=quadrature, points=points)
            d, se = item_contrast(fit, i)
            rows[i] = (d, se, fit["se_reason"])
        return rows

    def flags(rows):
        z = np.array([rows[i][0] / rows[i][1] if np.isfinite(rows[i][1]) else np.nan for i in testable])
        p = 2 * norm.sf(np.abs(z)); q = benjamini_hochberg(p)
        return z, p, q, {testable[j] for j in range(len(testable)) if np.isfinite(q[j]) and q[j] < alpha}

    history = []
    rows = test_items(list(range(k)))
    z, p, q, flagged = flags(rows)
    history.append({"Tur": 0, "Çapa sayısı": k - 1, "İşaretlenen": sorted(names[i] for i in flagged)})
    stable = True
    if purify:
        stable = False
        for r in range(1, max_rounds + 1):
            anchors = [j for j in range(k) if j not in flagged]
            if len(anchors) < min_anchors:
                warnings.append(f"Arındırma sonrası değişmez kabul edilen madde sayısı ({len(anchors)}) "
                                f"önerilen alt sınırın ({min_anchors}) altında; çapa kümesi kararsız.")
                break
            rows = test_items(anchors)
            z, p, q, new = flags(rows)
            history.append({"Tur": r, "Çapa sayısı": len(anchors), "İşaretlenen": sorted(names[i] for i in new)})
            if new == flagged:
                stable = True; break
            flagged = new
        if not stable:
            warnings.append("Çapa arındırma azami tur sayısında kararlı bir kümeye ulaşmadı; sonuçlar "
                            "çapa seçimine duyarlıdır.")
    final_anchors = [j for j in range(k) if j not in flagged]
    if len(final_anchors) < min_anchors:
        warnings.append(f"Değişmez madde sayısı ({len(final_anchors)}) önerilen alt sınırın ({min_anchors}) altında.")
    joint = fit_multigroup(x, obs, groups, [j for j in range(k) if j not in final_anchors],
                           quadrature=quadrature, points=points)
    table = []
    zi = dict(zip(testable, z)); pi = dict(zip(testable, p)); qi = dict(zip(testable, q))
    for i in range(k):
        if i in rows:
            d, se, why = rows[i]
            sig = bool(np.isfinite(qi[i]) and qi[i] < alpha)
            table.append({"Madde": names[i], "Δ (b_odak − b_ref)": d, "SH(Δ)": se, "z": zi[i],
                          "p": pi[i], "q (BH)": qi[i], "DIF işareti": sig,
                          "Etki sınıfı (ETS benzeri)": ets_class(d, se, sig),
                          "Çapa": i in final_anchors, "Not": why or ""})
        else:
            table.append({"Madde": names[i], "Δ (b_odak − b_ref)": np.nan, "SH(Δ)": np.nan, "z": np.nan,
                          "p": np.nan, "q (BH)": np.nan, "DIF işareti": False,
                          "Etki sınıfı (ETS benzeri)": "Test edilemedi", "Çapa": i in final_anchors,
                          "Not": untestable.get(i, "")})
    return {"table": pd.DataFrame(table), "anchors": [names[j] for j in final_anchors],
            "history": pd.DataFrame(history), "stable": stable, "joint": joint, "sizes": sizes,
            "reference": reference, "focal": focal, "alpha": alpha, "purify": purify,
            "min_anchors": min_anchors, "quadrature": quadrature, "points": points, "warnings": warnings,
            "estimand": "Δ = b_odak − b_referans (logit); Δ > 0: madde, aynı θ düzeyinde odak grup için daha zor.",
            "kept_rows": keep}
