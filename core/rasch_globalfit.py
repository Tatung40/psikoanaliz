"""Overall fit for the dichotomous Rasch model (B9).

Inferentially calibrated (asymptotic): limited-information M2 (Maydeu-Olivares & Joe,
2005/2006) from univariate and bivariate margins, valid for MML estimates and
complete data; df = s - q with s = k + k(k-1)/2 and q = k + 1 (b_1..b_k and the latent
SD); RMSEA2 and SRMSR. With missing responses M2 is not computed here.

Descriptive only (no reference distribution claimed): observed vs model-expected
raw-score distribution, item observed-expected proportions across WLE strata, and
standardized-residual summaries. Acceptable Infit/Outfit values are not evidence of
overall model fit.

grid="gauss-hermite" (production) or "mirt" (61 equally spaced nodes on [-6, 6]; used to
reproduce R mirt M2 exactly). free_sd=False treats the latent SD as known (the setting
under which mirt's M2 Jacobian, which omits group parameters, is comparable).
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm
from core.rasch import _grid, wle

CLASS_INFERENTIAL = "Çıkarımsal (asimptotik kalibreli)"
CLASS_DESCRIPTIVE = "Betimsel"


def _quadrature(sd, grid, points):
    if grid == "gauss-hermite":
        z, lw = _grid(points)
        return sd * z, np.exp(lw), z
    if grid == "mirt":
        theta = np.linspace(-6., 6., 61)
        w = norm.pdf(theta, 0., sd); w = w / w.sum()
        return theta, w, None
    raise ValueError("Bilinmeyen kuadratür ızgarası.")


def m2_statistic(data, b, sd, grid="gauss-hermite", points=41, free_sd=True, max_items=30):
    x = np.asarray(data, float)
    if not np.isfinite(x).all():
        return {"available": False, "reason": "M2 eksiksiz veri gerektirir; eksik yanıt bulunduğu için hesaplanmadı."}
    n, k = x.shape
    if k > max_items:
        return {"available": False, "reason": f"M2 en fazla {max_items} madde için hesaplanır (bellek sınırı)."}
    b = np.asarray(b, float)
    theta, w, z = _quadrature(float(sd), grid, points)
    P = expit(theta[:, None] - b[None, :])                                  # Q x k
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    sets = [(i,) for i in range(k)] + pairs
    s = len(sets)
    ind = np.zeros((s, k))
    for a, S in enumerate(sets):
        ind[a, list(S)] = 1.
    logP = np.log(P)
    m_q = np.exp(ind @ logP.T)                                              # s x Q moment integrands
    pi = m_q @ w
    obs = np.r_[x.mean(0), [np.mean(x[:, i] * x[:, j]) for i, j in pairs]]
    union = np.maximum(ind[:, None, :], ind[None, :, :]).reshape(s * s, k)
    pi_union = (np.exp(union @ logP.T) @ w).reshape(s, s)
    xi = pi_union - np.outer(pi, pi)
    # Jacobian of the margins: columns b_1..b_k (and the latent SD when free).
    q = k + (1 if free_sd else 0)
    delta = np.zeros((s, q))
    for j in range(k):
        delta[:, j] = -(m_q * ind[:, [j]] * (1 - P[:, j])[None, :]) @ w
    if free_sd:
        if grid == "gauss-hermite":
            dlog = (ind @ ((1 - P) * z[:, None]).T)                         # d log m / d sd
            delta[:, k] = (m_q * dlog) @ w
        else:
            v = theta ** 2 / sd ** 3
            delta[:, k] = (m_q * (v - w @ v)[None, :]) @ w
    xi_inv_d = np.linalg.solve(xi, delta)
    e = obs - pi
    xi_inv_e = np.linalg.solve(xi, e)
    middle = np.linalg.solve(delta.T @ xi_inv_d, delta.T @ xi_inv_e)
    m2 = float(n * (e @ xi_inv_e - (delta.T @ xi_inv_e) @ middle))
    df = s - q
    cov_obs = np.cov(x, rowvar=False, bias=True)
    model_cov = np.empty((k, k))
    pij = {pr: pi[k + a] for a, pr in enumerate(pairs)}
    for i in range(k):
        model_cov[i, i] = pi[i] * (1 - pi[i])
        for j in range(i + 1, k):
            model_cov[i, j] = model_cov[j, i] = pij[(i, j)] - pi[i] * pi[j]
    r_obs = cov_obs / np.sqrt(np.outer(np.diag(cov_obs), np.diag(cov_obs)))
    r_mod = model_cov / np.sqrt(np.outer(np.diag(model_cov), np.diag(model_cov)))
    tri = np.triu_indices(k, 1)
    srmsr = float(np.sqrt(np.mean((r_obs[tri] - r_mod[tri]) ** 2)))
    return {"available": True, "M2": m2, "df": int(df), "p": float(chi2.sf(m2, df)),
            "RMSEA2": float(np.sqrt(max(m2 - df, 0.) / (n * df))), "SRMSR": srmsr, "n": n, "k": k,
            "free_sd": bool(free_sd), "grid": grid, "reason": ""}


def raw_score_distribution(data, b, sd, points=41):
    """Observed vs model-expected raw-score frequencies (complete cases only; descriptive)."""
    x = np.asarray(data, float)
    complete = np.isfinite(x).all(axis=1)
    xc = x[complete]; n, k = xc.shape
    theta, w, _ = _quadrature(float(sd), "gauss-hermite", points)
    P = expit(theta[:, None] - np.asarray(b, float)[None, :])
    dist = np.zeros((len(theta), k + 1)); dist[:, 0] = 1.
    for j in range(k):                                                       # Lord-Wingersky
        new = np.zeros_like(dist)
        new[:, :-1] += dist[:, :-1] * (1 - P[:, [j]]); new[:, 1:] += dist[:, :-1] * P[:, [j]]
        dist = new
    expected = n * (w @ dist)
    observed = np.bincount(xc.sum(1).astype(int), minlength=k + 1)
    return pd.DataFrame({"Ham puan": np.arange(k + 1), "Gözlenen": observed, "Beklenen": expected,
                         "Gözlenen − Beklenen": observed - expected}), int(n), int((~complete).sum())


def item_strata(x, observed, theta, b, names, strata=5):
    """Observed vs expected proportion correct in WLE quantile strata (descriptive)."""
    cuts = np.unique(np.quantile(theta, np.linspace(0, 1, strata + 1)))
    group = np.clip(np.searchsorted(cuts, theta, side="right") - 1, 0, len(cuts) - 2)
    P = expit(theta[:, None] - np.asarray(b)[None, :])
    rows = []
    for g in range(len(cuts) - 1):
        m = group == g
        for i, name in enumerate(names):
            o = observed[m, i]
            if o.sum() == 0:
                continue
            obs_p = float(np.nanmean(x[m, i][o])); exp_p = float(P[m, i][o].mean())
            se = float(np.sqrt((P[m, i][o] * (1 - P[m, i][o])).sum()) / o.sum())
            rows.append({"Madde": name, "Tabaka": g + 1, "θ aralığı": f"[{cuts[g]:.2f}, {cuts[g + 1]:.2f}]",
                         "n": int(o.sum()), "Gözlenen p": obs_p, "Beklenen p": exp_p,
                         "Fark": obs_p - exp_p, "Standart fark (betimsel)": (obs_p - exp_p) / se if se > 0 else np.nan})
    return pd.DataFrame(rows)


def residual_summary(x, observed, theta, b):
    P = expit(np.asarray(theta)[:, None] - np.asarray(b)[None, :])
    z = ((np.nan_to_num(x) - P) / np.sqrt(P * (1 - P)))[observed]
    return {"Hücre": int(z.size), "Ortalama": float(z.mean()), "SS": float(z.std()),
            "Çarpıklık": float(((z - z.mean()) ** 3).mean() / z.std() ** 3),
            "|z| > 2 oranı": float((np.abs(z) > 2).mean()), "|z| > 3 oranı": float((np.abs(z) > 3).mean())}


def global_fit(analysis, points=41, strata=5):
    """Global fit block for a B8 `rasch_analysis` result (no re-estimation)."""
    f = analysis["fit"]
    x, obs, b, sd = f["data"], f["observed"], f["b"], f["sd"]
    theta = analysis["persons"]["WLE θ"].to_numpy(float)
    names = list(f["items"]["Madde"])
    m2 = m2_statistic(x, b, sd, points=points) if f["converged"] else \
        {"available": False, "reason": "Kestirim yakınsamadığı için M2 hesaplanmadı."}
    raw, n_complete, n_incomplete = raw_score_distribution(x, b, sd, points)
    classes = pd.DataFrame([
        {"İstatistik": "M2, RMSEA2", "Sınıf": CLASS_INFERENTIAL if m2["available"] else "Hesaplanmadı",
         "Not": "Asimptotik χ²; MML kestirimleri ve eksiksiz veri varsayılır." if m2["available"] else m2["reason"]},
        {"İstatistik": "SRMSR", "Sınıf": CLASS_DESCRIPTIVE, "Not": "Gözlenen ve model korelasyonlarının farkı."},
        {"İstatistik": "Ham puan dağılımı", "Sınıf": CLASS_DESCRIPTIVE,
         "Not": "Referans dağılım yok; yalnızca eksiksiz yanıtlı kişiler."},
        {"İstatistik": "Tabaka bazında gözlenen–beklenen", "Sınıf": CLASS_DESCRIPTIVE,
         "Not": "Standart farklar WLE tahmini ve tabakalama nedeniyle χ²/z dağılımına uymaz."},
        {"İstatistik": "Standart artık özeti", "Sınıf": CLASS_DESCRIPTIVE,
         "Not": "İkili artıklar normal dağılmaz; yalnızca betimseldir."},
        {"İstatistik": "Infit/Outfit (B8)", "Sınıf": CLASS_DESCRIPTIVE,
         "Not": "Kabul aralığındaki değerler genel model uyumunu kanıtlamaz."}])
    return {"m2": m2, "raw_scores": raw, "raw_complete_n": n_complete, "raw_incomplete_n": n_incomplete,
            "strata": item_strata(x, obs, theta, b, names, strata), "residuals": residual_summary(x, obs, theta, b),
            "classification": classes, "points": points, "strata_count": strata}
