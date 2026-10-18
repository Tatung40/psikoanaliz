"""Item and overall fit for the 1PL and 2PL models (B10).

Inferential (asymptotically calibrated): limited-information M2 (Maydeu-Olivares & Joe,
2005/2006) from univariate and bivariate margins, MML estimates, complete data. The
Jacobian contains exactly the freely estimated parameters of the fitted model:
1PL b_1..b_k (a = 1 and theta ~ N(0, 1) fixed), 2PL a_1..a_k, b_1..b_k (theta ~ N(0, 1)
fixed). df = k + k(k-1)/2 - q with q = k (1PL) or 2k (2PL); M2 is withheld when df < 1.
RMSEA2 (with N) is derived from M2; SRMSR is descriptive.

Descriptive only: observed vs expected raw-score distribution (Lord-Wingersky), item
observed vs expected proportions across ability strata (person estimates are data
based, so the standardized differences are not z-distributed), standardized-residual
summary. The pilot S-X² of core.irt / core.irt_itemfit are not changed here.

grid="gauss-hermite" (production) or "mirt" (61 equally spaced nodes on [-6, 6];
validation against R mirt).
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm
from core.rasch import _grid

CLASS_INFERENTIAL = "Çıkarımsal (asimptotik kalibreli)"
CLASS_DESCRIPTIVE = "Betimsel"
CLASS_EXPERIMENTAL = "Deneysel"


def quadrature(grid="gauss-hermite", points=41):
    if grid == "gauss-hermite":
        z, lw = _grid(points); return z, np.exp(lw)
    if grid == "mirt":
        theta = np.linspace(-6., 6., 61); w = norm.pdf(theta); return theta, w / w.sum()
    raise ValueError("Bilinmeyen kuadratür ızgarası.")


def m2_statistic(data, a, b, model, grid="gauss-hermite", points=41, max_items=30):
    x = np.asarray(data, float)
    if model not in ("1PL", "2PL"):
        raise ValueError("Bu M2 uygulaması yalnızca 1PL ve 2PL içindir.")
    if not np.isfinite(x).all():
        return {"available": False, "reason": "M2 eksiksiz veri gerektirir; eksik yanıt bulunduğu için hesaplanmadı."}
    n, k = x.shape
    if k > max_items:
        return {"available": False, "reason": f"M2 en fazla {max_items} madde için hesaplanır (bellek sınırı)."}
    q = k if model == "1PL" else 2 * k
    s = k + k * (k - 1) // 2
    if s - q < 1:
        return {"available": False, "reason": f"Serbestlik derecesi {s - q} < 1; M2 bu madde sayısında tanımsız."}
    a = np.ones(k) if model == "1PL" else np.asarray(a, float); b = np.asarray(b, float)
    theta, w = quadrature(grid, points)
    P = expit(a[None, :] * (theta[:, None] - b[None, :]))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    sets = [(i,) for i in range(k)] + pairs
    ind = np.zeros((s, k))
    for r, S in enumerate(sets):
        ind[r, list(S)] = 1.
    logP = np.log(P)
    m_q = np.exp(ind @ logP.T)
    pi = m_q @ w
    obs = np.r_[x.mean(0), [np.mean(x[:, i] * x[:, j]) for i, j in pairs]]
    union = np.maximum(ind[:, None, :], ind[None, :, :]).reshape(s * s, k)
    xi = (np.exp(union @ logP.T) @ w).reshape(s, s) - np.outer(pi, pi)
    delta = np.zeros((s, q))
    for j in range(k):
        dm = m_q * ind[:, [j]] * (1 - P[:, j])[None, :]                 # d m / d eta_j
        delta[:, j] = -(dm * a[j]) @ w                                  # d / d b_j
        if model == "2PL":
            delta[:, k + j] = (dm * (theta - b[j])[None, :]) @ w        # d / d a_j
    rank = np.linalg.matrix_rank(delta)
    if rank < q:
        return {"available": False, "reason": "Jacobian tam ranklı değil; M2 hesaplanamadı."}
    e = obs - pi
    xi_inv_e = np.linalg.solve(xi, e); xi_inv_d = np.linalg.solve(xi, delta)
    middle = np.linalg.solve(delta.T @ xi_inv_d, delta.T @ xi_inv_e)
    m2 = float(n * (e @ xi_inv_e - (delta.T @ xi_inv_e) @ middle))
    df = s - q
    cov_obs = np.cov(x, rowvar=False, bias=True)
    model_cov = np.empty((k, k))
    for i in range(k):
        model_cov[i, i] = pi[i] * (1 - pi[i])
    for r, (i, j) in enumerate(pairs):
        model_cov[i, j] = model_cov[j, i] = pi[k + r] - pi[i] * pi[j]
    sd_o = np.sqrt(np.diag(cov_obs)); sd_m = np.sqrt(np.diag(model_cov))
    tri = np.triu_indices(k, 1)
    srmsr = float(np.sqrt(np.mean((cov_obs[tri] / np.outer(sd_o, sd_o)[tri]
                                    - model_cov[tri] / np.outer(sd_m, sd_m)[tri]) ** 2)))
    return {"available": True, "M2": m2, "df": int(df), "p": float(chi2.sf(m2, df)),
            "RMSEA2": float(np.sqrt(max(m2 - df, 0.) / (n * df))), "SRMSR": srmsr, "n": n, "k": k,
            "free_parameters": q, "grid": grid, "jacobian": delta, "reason": ""}


def raw_score_distribution(data, a, b, points=41):
    x = np.asarray(data, float)
    complete = np.isfinite(x).all(axis=1); xc = x[complete]; n, k = xc.shape
    theta, w = quadrature("gauss-hermite", points)
    P = expit(np.asarray(a, float)[None, :] * (theta[:, None] - np.asarray(b, float)[None, :]))
    dist = np.zeros((len(theta), k + 1)); dist[:, 0] = 1.
    for j in range(k):
        new = np.zeros_like(dist)
        new[:, :-1] += dist[:, :-1] * (1 - P[:, [j]]); new[:, 1:] += dist[:, :-1] * P[:, [j]]
        dist = new
    expected = n * (w @ dist)
    observed = np.bincount(xc.sum(1).astype(int), minlength=k + 1)
    return pd.DataFrame({"Ham puan": np.arange(k + 1), "Gözlenen": observed, "Beklenen": expected,
                         "Gözlenen − Beklenen": observed - expected}), int(n), int((~complete).sum())


def item_strata(data, a, b, theta, names, strata=5):
    """Observed vs expected proportion correct per ability stratum (descriptive)."""
    x = np.asarray(data, float); obs = np.isfinite(x); theta = np.asarray(theta, float)
    cuts = np.unique(np.quantile(theta, np.linspace(0, 1, strata + 1)))
    grp = np.clip(np.searchsorted(cuts, theta, side="right") - 1, 0, len(cuts) - 2)
    P = expit(np.asarray(a, float)[None, :] * (theta[:, None] - np.asarray(b, float)[None, :]))
    rows = []
    for g in range(len(cuts) - 1):
        m = grp == g
        for i, name in enumerate(names):
            o = obs[m, i]
            if o.sum() == 0:
                continue
            po = float(x[m, i][o].mean()); pe = float(P[m, i][o].mean())
            se = float(np.sqrt((P[m, i][o] * (1 - P[m, i][o])).sum()) / o.sum())
            rows.append({"Madde": name, "Tabaka": g + 1, "θ aralığı": f"[{cuts[g]:.2f}, {cuts[g + 1]:.2f}]",
                         "Ortalama θ": float(theta[m][o].mean()), "n": int(o.sum()), "Gözlenen p": po,
                         "Beklenen p": pe, "Fark": po - pe,
                         "Standart fark (betimsel)": (po - pe) / se if se > 0 else np.nan})
    return pd.DataFrame(rows)


def residual_summary(data, a, b, theta):
    x = np.asarray(data, float); obs = np.isfinite(x)
    P = expit(np.asarray(a, float)[None, :] * (np.asarray(theta, float)[:, None] - np.asarray(b, float)[None, :]))
    z = ((np.nan_to_num(x) - P) / np.sqrt(P * (1 - P)))[obs]
    return {"Hücre": int(z.size), "Ortalama": float(z.mean()), "SS": float(z.std()),
            "|z| > 2 oranı": float((np.abs(z) > 2).mean()), "|z| > 3 oranı": float((np.abs(z) > 3).mean())}


def global_fit(data, a, b, model, theta, theta_method, converged=True, points=41, strata=5):
    """Fit block for a fitted 1PL/2PL table (no re-estimation)."""
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(np.shape(data)[1])]
    a = np.ones(len(names)) if model == "1PL" else np.asarray(a, float)
    m2 = (m2_statistic(data, a, b, model, points=points) if converged else
          {"available": False, "reason": "Kestirim yakınsamadığı için M2 hesaplanmadı."})
    raw, n_c, n_i = raw_score_distribution(data, a, b, points)
    classes = pd.DataFrame([
        {"İstatistik": "M2, p, RMSEA2", "Sınıf": CLASS_INFERENTIAL if m2["available"] else "Hesaplanmadı",
         "Not": (f"Asimptotik χ²; sd = s − q, q = {m2['free_parameters']} serbest madde parametresi; MML ve eksiksiz "
                 "veri varsayılır." if m2["available"] else m2["reason"])},
        {"İstatistik": "SRMSR", "Sınıf": CLASS_DESCRIPTIVE, "Not": "Gözlenen ve model madde korelasyonlarının farkı."},
        {"İstatistik": "Ham puan dağılımı", "Sınıf": CLASS_DESCRIPTIVE, "Not": "Referans dağılım yok."},
        {"İstatistik": "Tabaka bazında gözlenen–beklenen", "Sınıf": CLASS_DESCRIPTIVE,
         "Not": f"{theta_method} θ ile tabakalama; standart farklar z dağılımına uymaz."},
        {"İstatistik": "Standart artık özeti", "Sınıf": CLASS_DESCRIPTIVE, "Not": "İkili artıklar normal dağılmaz."},
        {"İstatistik": "S-X² (mevcut sekme)", "Sınıf": CLASS_EXPERIMENTAL,
         "Not": "Sabit N(0,1) ve kestirilmiş madde parametreleriyle; B10'da yeniden doğrulanmadı."}])
    return {"m2": m2, "raw_scores": raw, "raw_complete_n": n_c, "raw_incomplete_n": n_i,
            "strata": item_strata(data, a, b, theta, names, strata), "residuals": residual_summary(data, a, b, theta),
            "classification": classes, "theta_method": theta_method, "points": points, "strata_count": strata,
            "model": model}
