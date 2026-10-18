"""Dichotomous Rasch model: MML calibration, WLE persons, Infit/Outfit, reliability.

Independent implementation (B8). Model: P(X_ni = 1) = logistic(theta_n - b_i),
theta ~ N(0, sigma^2). Identification: latent mean fixed at 0, latent SD and
all item difficulties free (same convention as R mirt itemtype='Rasch').

Estimation is marginal maximum likelihood: the marginal log-likelihood is
maximized directly (L-BFGS-B, analytic score) on a Gauss-Hermite grid with
theta = sigma * z. Missing responses (NaN) are skipped in every likelihood,
i.e. ignorable missingness (MAR) is assumed. Convergence is decided from the
score norm, not from the optimizer flag.

Person locations: Warm (1989) weighted likelihood estimates, finite for
extreme scores. Fit statistics follow Wright & Masters (1982): residual
y = x - P, variance W = P(1-P), kurtosis C = P(1-P)^4 + (1-P)P^4;
Outfit = mean(y^2/W), Infit = sum(y^2)/sum(W); standardized with the
Wilson-Hilferty transformation, q_out^2 = sum(C/W^2)/n^2 - 1/n,
q_in^2 = sum(C - W^2)/(sum W)^2. q is not truncated (mirt truncates at sqrt 2).
"""
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, log_expit

B_BOUND = 10.
LOG_SD_BOUNDS = (np.log(1e-3), np.log(20.))
SCORE_TOLERANCE = 1e-6           # max |d logL / d param| per person at convergence


def _as_matrix(data):
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("Rasch analizi iki boyutlu kişi × madde verisi gerektirir.")
    observed = np.isfinite(x)
    if not np.isin(x[observed], [0., 1.]).all():
        raise ValueError("Rasch (ikili) analizi yalnızca 0/1 ve eksik (boş) değer kabul eder.")
    return x, observed


def validate_rasch_data(data, min_persons=30, min_items=3):
    """Return (x, observed, kept_rows, report); persons with no response are dropped."""
    x, observed = _as_matrix(data)
    keep = observed.any(axis=1)
    x, observed = x[keep], observed[keep]
    n, k = x.shape
    if n < min_persons or k < min_items:
        raise ValueError(f"Rasch analizi için en az {min_persons} katılımcı ve {min_items} madde gereklidir.")
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(k)]
    counts = observed.sum(axis=0)
    correct = np.where(observed, x, 0.).sum(axis=0)
    bad = [names[i] for i in range(k) if counts[i] == 0 or correct[i] == 0 or correct[i] == counts[i]]
    if bad:
        raise ValueError("Şu maddelerde gözlenen yanıtların tamamı aynı (veya hiç yanıt yok); Rasch "
                         "güçlüğü sonlu kestirilemez: " + ", ".join(map(str, bad)))
    report = {"dropped_empty_persons": int((~keep).sum()), "missing_cells": int((~observed).sum()),
              "missing_rate": float((~observed).mean()), "item_names": names}
    return x, observed, keep, report


def _grid(points):
    gh, w = hermgauss(int(points))
    return np.sqrt(2.) * gh, np.log(w / np.sqrt(np.pi))


def _person_loglik_on_grid(x1, x0, b, sd, z):
    """N x Q log-likelihood of each response pattern at theta = sd * z."""
    eta = sd * z[:, None] - b[None, :]
    return x1 @ log_expit(eta).T + x0 @ log_expit(-eta).T


def marginal_loglik(data, b, sd, points=41):
    x, obs = _as_matrix(data)
    z, lw = _grid(points)
    x1 = np.where(obs, x, 0.); x0 = np.where(obs, 1. - x, 0.)
    return float(logsumexp(_person_loglik_on_grid(x1, x0, np.asarray(b, float), float(sd), z) + lw, axis=1).sum())


def _objective(params, x1, x0, z, lw, n):
    b, log_sd = params[:-1], params[-1]
    sd = np.exp(log_sd)
    lp = _person_loglik_on_grid(x1, x0, b, sd, z) + lw
    norm = logsumexp(lp, axis=1)
    post = np.exp(lp - norm[:, None])                         # N x Q
    p = expit(sd * z[:, None] - b[None, :])                    # Q x k
    obs = x1 + x0
    # d/d eta_qi summed over persons: sum_n post_nq (x1_ni - obs_ni * p_qi)
    resid = post.T @ x1 - (post.T @ obs) * p                    # Q x k
    grad_b = -resid.sum(axis=0)
    grad_log_sd = float((resid * (sd * z)[:, None]).sum())
    value = -float(norm.sum()) / n
    return value, -np.r_[grad_b, grad_log_sd] / n


def _score(params, x1, x0, z, lw):
    return -_objective(params, x1, x0, z, lw, 1.)[1]


def _information(params, x1, x0, z, lw):
    """Observed information: central-difference Jacobian of the analytic score."""
    m = len(params); hess = np.zeros((m, m))
    for j in range(m):
        h = 1e-5 * max(1., abs(params[j]))
        up, dn = params.copy(), params.copy(); up[j] += h; dn[j] -= h
        hess[:, j] = -(_score(up, x1, x0, z, lw) - _score(dn, x1, x0, z, lw)) / (2 * h)
    return (hess + hess.T) / 2


def fit_rasch(data, points=41, max_iter=2000, score_tolerance=SCORE_TOLERANCE):
    """MML calibration of the dichotomous Rasch model (latent mean 0, SD free)."""
    x, obs, keep, report = validate_rasch_data(data)
    n, k = x.shape
    z, lw = _grid(points)
    x1 = np.where(obs, x, 0.); x0 = np.where(obs, 1. - x, 0.)
    p = np.clip(x1.sum(0) / obs.sum(0), .01, .99)
    start = np.r_[np.clip(-np.log(p / (1 - p)), -4, 4), 0.]
    bounds = [(-B_BOUND, B_BOUND)] * k + [LOG_SD_BOUNDS]
    res = minimize(_objective, start, args=(x1, x0, z, lw, n), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 1e-12})
    params = np.asarray(res.x, float)
    score = _score(params, x1, x0, z, lw) / n
    at_bound = np.r_[np.abs(np.abs(params[:-1]) - B_BOUND) < 1e-6,
                     min(abs(params[-1] - LOG_SD_BOUNDS[0]), abs(params[-1] - LOG_SD_BOUNDS[1])) < 1e-6]
    max_score = float(np.abs(score[~at_bound]).max()) if (~at_bound).any() else 0.
    converged = bool(np.isfinite(params).all() and max_score <= score_tolerance and not at_bound.any())
    b, sd = params[:-1], float(np.exp(params[-1]))
    ll = marginal_loglik(x, b, sd, points)
    # Standard errors from the observed information, refused unless clearly positive definite.
    se_b = np.full(k, np.nan); se_sd = np.nan; se_reason = ""
    if not converged:
        se_reason = "Kestirim yakınsamadığı için standart hata raporlanmaz."
    else:
        info = _information(params, x1, x0, z, lw)
        eig = np.linalg.eigvalsh(info)
        if not np.isfinite(eig).all() or eig[0] <= 1e-8 * max(eig[-1], 1.):
            se_reason = "Gözlenen bilgi matrisi pozitif tanımlı değil; SH üretilemez."
        else:
            cov = np.linalg.inv(info)
            se_b = np.sqrt(np.diag(cov)[:-1]); se_sd = float(sd * np.sqrt(cov[-1, -1]))
    items = pd.DataFrame({"Madde": report["item_names"], "b": b, "SH(b)": se_b,
                          "Yanıt sayısı": obs.sum(0).astype(int),
                          "Doğru oranı": x1.sum(0) / obs.sum(0)})
    warnings = []
    if report["missing_cells"]:
        warnings.append(f"Verinin %{100 * report['missing_rate']:.1f}'i eksik. Eksik yanıtlar olabilirlikten "
                        "çıkarıldı; bu yaklaşım eksikliğin rastgele (MAR) olduğunu varsayar.")
    if report["dropped_empty_persons"]:
        warnings.append(f"Hiç yanıtı olmayan {report['dropped_empty_persons']} kişi analizden çıkarıldı.")
    if not converged:
        warnings.append("MML kestirimi yakınsama ölçütünü sağlamadı (skor normu "
                        f"{max_score:.2g}/kişi, eşik {score_tolerance:g}); sonuçlar geçicidir.")
    return {"model": "Rasch", "items": items, "b": b, "sd": sd, "se_sd": se_sd, "se_reason": se_reason,
            "loglik": ll, "aic": 2 * (k + 1) - 2 * ll, "bic": np.log(n) * (k + 1) - 2 * ll,
            "n": n, "k": k, "param_count": k + 1, "points": int(points), "converged": converged,
            "max_score_per_person": max_score, "score_tolerance": float(score_tolerance),
            "optimizer_success": bool(res.success), "optimizer_message": str(res.message),
            "iterations": int(res.nit), "at_bound": bool(at_bound.any()),
            "data": x, "observed": obs, "kept_rows": keep, "warnings": warnings, **{
                "missing_cells": report["missing_cells"], "missing_rate": report["missing_rate"],
                "dropped_empty_persons": report["dropped_empty_persons"]}}


def wle(x, observed, b, lo=-12., hi=12., iterations=80):
    """Warm WLE per person (bisection on the weighted score; monotone for Rasch)."""
    x = np.asarray(x, float); observed = np.asarray(observed, bool); b = np.asarray(b, float)
    r = np.where(observed, x, 0.).sum(axis=1)
    def weighted_score(theta):
        p = expit(theta[:, None] - b[None, :])
        w = np.where(observed, p * (1 - p), 0.)
        info = w.sum(1); j = np.where(observed, w * (1 - 2 * p), 0.).sum(1)
        return r - np.where(observed, p, 0.).sum(1) + j / (2 * info)
    low = np.full(len(x), lo); high = np.full(len(x), hi)
    for _ in range(iterations):
        mid = (low + high) / 2
        positive = weighted_score(mid) > 0
        low = np.where(positive, mid, low); high = np.where(positive, high, mid)
    theta = (low + high) / 2
    p = expit(theta[:, None] - b[None, :])
    info = np.where(observed, p * (1 - p), 0.).sum(1)
    return theta, 1 / np.sqrt(info)


def eap(x, observed, b, sd, points=41):
    z, lw = _grid(points)
    x1 = np.where(observed, x, 0.); x0 = np.where(observed, 1. - x, 0.)
    lp = _person_loglik_on_grid(x1, x0, np.asarray(b, float), float(sd), z) + lw
    post = np.exp(lp - logsumexp(lp, axis=1)[:, None])
    theta = sd * z
    mean = post @ theta
    return mean, np.sqrt(np.maximum(post @ theta ** 2 - mean ** 2, 0.))


def _zstd(ms, q):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.cbrt(ms) - 1) * (3 / q) + q / 3


def fit_statistics(x, observed, theta, b, use=None):
    """Item and person Infit/Outfit MNSQ and ZSTD; `use` selects persons for item fit."""
    x = np.asarray(x, float); observed = np.asarray(observed, bool)
    p = expit(np.asarray(theta, float)[:, None] - np.asarray(b, float)[None, :])
    w = p * (1 - p); c = p * (1 - p) ** 4 + (1 - p) * p ** 4
    y2 = np.where(observed, (np.nan_to_num(x) - p) ** 2, 0.)
    wm = np.where(observed, w, 0.); cm = np.where(observed, c, 0.)
    def stats(mask, axis):
        y2m, wmm, cmm = y2 * mask, wm * mask, cm * mask
        n = mask.sum(axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            outfit = (np.where(mask, y2 / w, 0.)).sum(axis) / n
            infit = y2m.sum(axis) / wmm.sum(axis)
            q_out = np.sqrt(np.where(mask, c / w ** 2, 0.).sum(axis) / n ** 2 - 1 / n)
            q_in = np.sqrt((cmm - wmm ** 2 * mask).sum(axis) / wmm.sum(axis) ** 2)
        return infit, _zstd(infit, q_in), outfit, _zstd(outfit, q_out), n
    use = np.ones(len(x), bool) if use is None else np.asarray(use, bool)
    item = stats(observed & use[:, None], 0)
    person = stats(observed, 1)
    keys = ("Infit MNSQ", "Infit ZSTD", "Outfit MNSQ", "Outfit ZSTD", "Gözlem")
    return dict(zip(keys, item)), dict(zip(keys, person))


def separation(estimates, se):
    """Model-SE separation reliability R = (Var - mean SE^2)/Var, G = sqrt(R/(1-R)).

    Var uses ddof=0 (observed variance of the estimates); R is truncated at 0.
    """
    est = np.asarray(estimates, float); se = np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se)
    var = float(np.var(est[ok])); mse = float(np.mean(se[ok] ** 2))
    true_var = max(var - mse, 0.)
    rel = true_var / var if var > 0 else np.nan
    g = np.sqrt(true_var / mse) if mse > 0 else np.nan
    return {"Gözlenen varyans": var, "Ortalama hata varyansı": mse, "Güvenirlik": rel,
            "Ayırma indeksi (G)": g, "Tabaka sayısı": (4 * g + 1) / 3, "Kullanılan": int(ok.sum())}


def rasch_analysis(data, points=41, mnsq_range=(0.7, 1.3), zstd_limit=2.0,
                   include_extreme_in_item_fit=False):
    """Full analysis: calibration, persons, fit, reliability and Wright map data.

    Default MNSQ range 0.7-1.3 (Wright & Linacre, 1994, multiple-choice tests); an item or
    person is flagged only if MNSQ is outside the range AND |ZSTD| > zstd_limit.
    """
    fit = fit_rasch(data, points=points)
    x, obs, b = fit["data"], fit["observed"], fit["b"]
    theta, se = wle(x, obs, b)
    eap_mean, eap_sd = eap(x, obs, b, fit["sd"], points)
    raw = np.where(obs, x, 0.).sum(1); maximum = obs.sum(1)
    extreme = (raw == 0) | (raw == maximum)
    item_fit, person_fit = fit_statistics(x, obs, theta, b,
                                          use=None if include_extreme_in_item_fit else ~extreme)
    lo, hi = mnsq_range
    def flag(i, o, zi, zo):
        out = ((i < lo) | (i > hi) | (o < lo) | (o > hi)) & ((np.abs(zi) > zstd_limit) | (np.abs(zo) > zstd_limit))
        return np.where(np.isfinite(i), np.where(out, "İnceleme önerilir", "Kabul aralığında"), "Hesaplanmadı")
    items = fit["items"].copy()
    for key in ("Infit MNSQ", "Infit ZSTD", "Outfit MNSQ", "Outfit ZSTD"):
        items[key] = item_fit[key]
    items["Uyum"] = flag(item_fit["Infit MNSQ"], item_fit["Outfit MNSQ"], item_fit["Infit ZSTD"], item_fit["Outfit ZSTD"])
    persons = pd.DataFrame({"Kişi": np.arange(1, len(x) + 1), "Ham puan": raw.astype(int),
                            "Yanıtlanan madde": maximum.astype(int), "WLE θ": theta, "SH(θ)": se,
                            "EAP θ": eap_mean, "Posterior SS": eap_sd, "Uç puan": extreme})
    for key in ("Infit MNSQ", "Infit ZSTD", "Outfit MNSQ", "Outfit ZSTD"):
        persons[key] = np.where(extreme, np.nan, person_fit[key])
    persons["Uyum"] = np.where(extreme, "Uç puan (uyum hesaplanmaz)",
                               flag(persons["Infit MNSQ"].to_numpy(), persons["Outfit MNSQ"].to_numpy(),
                                    persons["Infit ZSTD"].to_numpy(), persons["Outfit ZSTD"].to_numpy()))
    person_rel = separation(theta[~extreme], se[~extreme])
    item_rel = separation(b, fit["items"]["SH(b)"].to_numpy())
    marginal = float(np.var(eap_mean) / (np.var(eap_mean) + np.mean(eap_sd ** 2)))
    reliability = pd.DataFrame([{"Gösterge": "Kişi (WLE, model SH, uç puanlar hariç)", **person_rel},
                                {"Gösterge": "Madde (MML SH)", **item_rel}])
    warnings = list(fit["warnings"])
    if extreme.any():
        warnings.append(f"{int(extreme.sum())} kişinin puanı uç değerde (0 veya tam puan); WLE ile sonlu "
                        "kestirildi, uyum istatistikleri ve kişi güvenirliği hesabına katılmadı.")
    if fit["se_reason"]:
        warnings.append(fit["se_reason"] + " Madde güvenirliği hesaplanamaz.")
    return {"fit": fit, "items": items, "persons": persons, "reliability": reliability,
            "marginal_reliability": marginal, "extreme": extreme, "warnings": warnings,
            "settings": {"points": int(points), "mnsq_range": tuple(mnsq_range), "zstd_limit": float(zstd_limit),
                         "include_extreme_in_item_fit": bool(include_extreme_in_item_fit)},
            "wright": {"persons": theta, "items": pd.Series(b, index=fit["items"]["Madde"])}}
