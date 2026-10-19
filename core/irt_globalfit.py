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


FREE = {"1PL": ("b",), "2PL": ("b", "a"), "3PL": ("b", "a", "c"), "4PL": ("b", "a", "c", "d")}


def m2_statistic(data, a, b, model, grid="gauss-hermite", points=41, max_items=30, c=None, d=None, fixed=None,
                 max_condition=1e10):
    """M2 with a Jacobian over the model's free parameters (columns: b, a, c, d blocks).

    `fixed` (optional dict name -> boolean array) marks parameters on an estimation bound; their
    columns are removed (they are treated as fixed) and the result is flagged
    `conditional=True` -- standard asymptotics do not apply automatically in that case.
    """
    x = np.asarray(data, float)
    if model not in FREE:
        raise ValueError("Geçersiz model.")
    if not np.isfinite(x).all():
        return {"available": False, "reason": "M2 eksiksiz veri gerektirir; eksik yanıt bulunduğu için hesaplanmadı."}
    n, k = x.shape
    if k > max_items:
        return {"available": False, "reason": f"M2 en fazla {max_items} madde için hesaplanır (bellek sınırı)."}
    keep = {p: np.ones(k, bool) if not fixed or p not in fixed else ~np.asarray(fixed[p], bool) for p in FREE[model]}
    q = int(sum(v.sum() for v in keep.values()))
    s = k + k * (k - 1) // 2
    if s - q < 1:
        return {"available": False, "reason": f"Serbestlik derecesi {s - q} < 1; M2 bu madde sayısında tanımsız "
                                              f"(s = {s} moment, q = {q} serbest parametre)."}
    a = np.ones(k) if model == "1PL" else np.asarray(a, float); b = np.asarray(b, float)
    c = np.zeros(k) if model in ("1PL", "2PL") or c is None else np.asarray(c, float)
    d = np.ones(k) if model != "4PL" or d is None else np.asarray(d, float)
    theta, w = quadrature(grid, points)
    L = expit(a[None, :] * (theta[:, None] - b[None, :]))
    P = c[None, :] + (d - c)[None, :] * L
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
    blocks = []
    for name in FREE[model]:
        col = np.zeros((s, k))
        for j in range(k):
            mP = m_q * ind[:, [j]] / P[:, j][None, :]                     # d m / d P_j
            dP = {"b": -(d[j] - c[j]) * a[j] * L[:, j] * (1 - L[:, j]),
                  "a": (d[j] - c[j]) * (theta - b[j]) * L[:, j] * (1 - L[:, j]),
                  "c": 1 - L[:, j], "d": L[:, j]}[name]
            col[:, j] = (mP * dP[None, :]) @ w
        blocks.append(col[:, keep[name]])
    delta = np.hstack(blocks)
    sv = np.linalg.svd(delta, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    if rank < q:
        return {"available": False, "reason": f"Jacobian tam ranklı değil (rank {rank} < {q}); M2 hesaplanamadı."}
    e = obs - pi
    xi_inv_e = np.linalg.solve(xi, e); xi_inv_d = np.linalg.solve(xi, delta)
    dtd = delta.T @ xi_inv_d
    cond = float(np.linalg.cond(dtd))
    if not np.isfinite(cond) or cond > max_condition:
        return {"available": False, "reason": f"Δ'Ξ⁻¹Δ kötü koşullu (koşul sayısı {cond:.2g}); M2 hesaplanamadı."}
    middle = np.linalg.solve(dtd, delta.T @ xi_inv_e)
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
            "free_parameters": q, "grid": grid, "jacobian": delta, "reason": "", "condition": cond,
            "conditional": bool(q < k * len(FREE[model])), "fixed_parameters": int(k * len(FREE[model]) - q)}


def _irf(theta, a, b, c=None, d=None):
    L = expit(np.asarray(a, float)[None, :] * (np.asarray(theta, float)[:, None] - np.asarray(b, float)[None, :]))
    if c is None and d is None:
        return L
    c = np.zeros(L.shape[1]) if c is None else np.asarray(c, float)
    d = np.ones(L.shape[1]) if d is None else np.asarray(d, float)
    return c[None, :] + (d - c)[None, :] * L


def raw_score_distribution(data, a, b, points=41, c=None, d=None):
    x = np.asarray(data, float)
    complete = np.isfinite(x).all(axis=1); xc = x[complete]; n, k = xc.shape
    theta, w = quadrature("gauss-hermite", points)
    P = _irf(theta, a, b, c, d)
    dist = np.zeros((len(theta), k + 1)); dist[:, 0] = 1.
    for j in range(k):
        new = np.zeros_like(dist)
        new[:, :-1] += dist[:, :-1] * (1 - P[:, [j]]); new[:, 1:] += dist[:, :-1] * P[:, [j]]
        dist = new
    expected = n * (w @ dist)
    observed = np.bincount(xc.sum(1).astype(int), minlength=k + 1)
    return pd.DataFrame({"Ham puan": np.arange(k + 1), "Gözlenen": observed, "Beklenen": expected,
                         "Gözlenen − Beklenen": observed - expected}), int(n), int((~complete).sum())


def item_strata(data, a, b, theta, names, strata=5, c=None, d=None):
    """Observed vs expected proportion correct per ability stratum (descriptive)."""
    x = np.asarray(data, float); obs = np.isfinite(x); theta = np.asarray(theta, float)
    ok = np.isfinite(theta)                                 # undefined person estimates are excluded
    x, obs, theta = x[ok], obs[ok], theta[ok]
    cuts = np.unique(np.quantile(theta, np.linspace(0, 1, strata + 1)))
    grp = np.clip(np.searchsorted(cuts, theta, side="right") - 1, 0, len(cuts) - 2)
    P = _irf(theta, a, b, c, d)
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


def residual_summary(data, a, b, theta, c=None, d=None):
    theta = np.asarray(theta, float); ok = np.isfinite(theta)
    x = np.asarray(data, float)[ok]; obs = np.isfinite(x); theta = theta[ok]
    P = _irf(theta, a, b, c, d)
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


def global_fit_34(data, a, b, c, d, model, theta, theta_method, converged=True, fixed=None, points=41, strata=5):
    """Fit block for a fitted 3PL/4PL table (B11; no re-estimation).

    M2 is inferential only for a converged, fully interior solution (no parameter on an
    estimation bound) with a full-rank, well-conditioned Jacobian. With bound-active
    parameters their columns are dropped (held fixed) and M2/df are shown as experimental
    without a p-value: the asymptotic chi-square reference is not established at the boundary.
    """
    if model not in ("3PL", "4PL"):
        raise ValueError("global_fit_34 yalnızca 3PL ve 4PL içindir.")
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(np.shape(data)[1])]
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    d = np.ones(len(a)) if model == "3PL" else np.asarray(d, float)
    if not converged:
        m2 = {"available": False, "reason": "Kestirim yakınsamadığı için M2 hesaplanmadı."}
    else:
        m2 = m2_statistic(data, a, b, model, points=points, c=c, d=d, fixed=fixed)
    if m2["available"] and m2["conditional"]:
        m2 = dict(m2, p_reported=np.nan, RMSEA2_reported=np.nan)
        cls, note = CLASS_EXPERIMENTAL, (f"{m2['fixed_parameters']} parametre kestirim sınırında; bu parametreler sabit "
                                          f"kabul edilerek q = {m2['free_parameters']} ile hesaplandı. Sınırda χ² "
                                          "referansı geçerli olmadığından p ve RMSEA2 verilmez.")
    elif m2["available"]:
        m2 = dict(m2, p_reported=m2["p"], RMSEA2_reported=m2["RMSEA2"])
        cls, note = CLASS_INFERENTIAL, (f"Asimptotik χ²; sd = s − q, q = {m2['free_parameters']} serbest madde "
                                        "parametresi (c" + (", d" if model == "4PL" else "") + " dahil); tüm "
                                        "parametreler iç noktada; eksiksiz veri.")
    else:
        cls, note = "Hesaplanmadı", m2["reason"]
    raw, n_c, n_i = raw_score_distribution(data, a, b, points, c, d)
    classes = pd.DataFrame([
        {"İstatistik": "M2, p, RMSEA2", "Sınıf": cls, "Not": note},
        {"İstatistik": "SRMSR", "Sınıf": CLASS_DESCRIPTIVE, "Not": "Gözlenen ve model madde korelasyonlarının farkı."},
        {"İstatistik": "Ham puan dağılımı", "Sınıf": CLASS_DESCRIPTIVE, "Not": "Referans dağılım yok."},
        {"İstatistik": "Tabaka bazında gözlenen–beklenen", "Sınıf": CLASS_DESCRIPTIVE,
         "Not": f"{theta_method} θ ile tabakalama; standart farklar z dağılımına uymaz."},
        {"İstatistik": "Standart artık özeti", "Sınıf": CLASS_DESCRIPTIVE, "Not": "İkili artıklar normal dağılmaz."},
        {"İstatistik": "S-X² (mevcut sekme)", "Sınıf": CLASS_EXPERIMENTAL,
         "Not": "Sabit N(0,1) ile; B11'de 3PL/4PL için yeniden doğrulanmadı."}])
    return {"m2": m2, "raw_scores": raw, "raw_complete_n": n_c, "raw_incomplete_n": n_i,
            "strata": item_strata(data, a, b, theta, names, strata, c, d),
            "residuals": residual_summary(data, a, b, theta, c, d), "classification": classes,
            "theta_method": theta_method, "points": points, "strata_count": strata, "model": model}
