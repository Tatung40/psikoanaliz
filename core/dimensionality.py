"""Boyutluluk analizleri: Pearson / tetrakorik / polikorik korelasyon, KMO, Bartlett,
paralel analiz (Horn, sütun permütasyonu), temel eksen faktörleştirme (PAF) ve
varimax/promax döndürme.

Polikorik/tetrakorik korelasyon: iki aşamalı ML (eşikler marjinallerden, rho
ikili normal olabilirlik maksimizasyonundan). İkili normal CDF, Gauss-Legendre
integrali ile hesaplanır ve testlerde scipy.stats.multivariate_normal ile
bağımsız olarak karşılaştırılır.

Evrensel kesme puanı kullanılmaz; tek boyutluluk özeti betimseldir ve karar
paralel analize dayanır.
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize

_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)
RHO_MAX = 0.9999
MAX_CATEGORIES = 10


def bvn_cdf(h, k, rho):
    """P(X<=h, Y<=k) for standard bivariate normal (vectorised in h, k)."""
    h = np.asarray(h, float); k = np.asarray(k, float)
    base = stats.norm.cdf(h) * stats.norm.cdf(k)
    if rho == 0:
        return base
    lim = np.arcsin(rho)                                # r = sin(t): no endpoint singularity
    t = 0.5 * lim * (_GL_X + 1.0)
    w = 0.5 * lim * _GL_W
    hh, kk = h[..., None], k[..., None]
    st, ct2 = np.sin(t), np.cos(t) ** 2
    f = np.exp(-(hh * hh - 2 * st * hh * kk + kk * kk) / (2 * ct2))
    return base + (f * w).sum(-1) / (2 * np.pi)


def data_type(df):
    """'binary', 'ordinal' (2 < kategori <= 10, tamsayı) veya 'continuous'."""
    x = df.to_numpy(float)
    vals = np.unique(x[np.isfinite(x)])
    if len(vals) <= 2:
        return "binary"
    if len(vals) <= MAX_CATEGORIES and np.allclose(vals, np.round(vals)):
        return "ordinal"
    return "continuous"


def _thresholds(v, cats):
    counts = np.array([(v == c).sum() for c in cats], float)
    cum = np.cumsum(counts)[:-1] / counts.sum()
    return np.r_[-np.inf, stats.norm.ppf(cum), np.inf]


def polychoric_pair(x, y):
    """İki aşamalı ML polikorik (2 kategori için tetrakorik) korelasyon.

    Returns (rho, n, flag). flag: '' | 'sabit madde' | 'sınır' .
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    cx, cy = np.unique(x), np.unique(y)
    if len(cx) < 2 or len(cy) < 2:
        return np.nan, int(m.sum()), "sabit madde"
    tx, ty = _thresholds(x, cx), _thresholds(y, cy)
    table = np.zeros((len(cx), len(cy)))
    np.add.at(table, (np.searchsorted(cx, x), np.searchsorted(cy, y)), 1)
    zero = bool((table == 0).any())
    if zero:                       # psych::tetrachoric/polychoric correct=0.5 ile aynı
        table = np.where(table == 0, 0.5, table)
    a = np.clip(tx, -8.5, 8.5); b = np.clip(ty, -8.5, 8.5)
    A, B = np.meshgrid(a, b, indexing="ij")

    def nll(rho):
        F = bvn_cdf(A, B, rho)
        p = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
        return -float((table * np.log(np.clip(p, 1e-300, None))).sum())

    res = optimize.minimize_scalar(nll, bounds=(-RHO_MAX, RHO_MAX), method="bounded",
                                   options={"xatol": 1e-7})
    rho = float(res.x)
    flag = "; ".join(f for f in ("boş hücre (0.5 düzeltmesi)" if zero else "",
                                     "sınır" if abs(rho) > 0.995 else "") if f)
    return rho, int(m.sum()), flag


def correlation_matrix(df, method="auto"):
    """method: auto | pearson | tetrachoric | polychoric.

    Returns dict(R, method, n_pairs, flags, smoothed, min_eigen_before).
    Polikorik/tetrakorik matris pozitif tanımlı değilse özdeğer kırpma ile
    düzeltilir ve bu durum raporlanır.
    """
    kind = data_type(df)
    if method == "auto":
        method = {"binary": "tetrachoric", "ordinal": "polychoric", "continuous": "pearson"}[kind]
    if method == "tetrachoric" and kind != "binary":
        raise ValueError("Tetrakorik korelasyon yalnızca iki kategorili veri içindir.")
    if method == "polychoric" and kind == "continuous":
        raise ValueError("Polikorik korelasyon sürekli veri için uygun değildir.")
    cols = list(df.columns)
    x = df.to_numpy(float)
    k = x.shape[1]
    R = np.eye(k); N = np.zeros((k, k), int); flags = []
    for i in range(k):
        N[i, i] = int(np.isfinite(x[:, i]).sum())
        for j in range(i + 1, k):
            if method == "pearson":
                m = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
                r = np.corrcoef(x[m, i], x[m, j])[0, 1] if m.sum() > 2 else np.nan
                n, flag = int(m.sum()), ""
            else:
                r, n, flag = polychoric_pair(x[:, i], x[:, j])
            R[i, j] = R[j, i] = r; N[i, j] = N[j, i] = n
            if flag:
                flags.append({"Madde 1": cols[i], "Madde 2": cols[j], "Uyarı": flag})
    if np.isnan(R).any():
        raise ValueError("Tanımsız korelasyon var (sabit madde veya yetersiz ortak yanıt).")
    ev = np.linalg.eigvalsh(R)
    smoothed = False
    if ev.min() <= 1e-8:
        R = smooth_pd(R); smoothed = True
    return {"R": pd.DataFrame(R, index=cols, columns=cols), "method": method, "data_type": kind,
            "n_pairs": pd.DataFrame(N, index=cols, columns=cols), "flags": pd.DataFrame(flags),
            "smoothed": smoothed, "min_eigen_before": float(ev.min())}


def smooth_pd(R, eps=1e-6):
    w, v = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    S = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    np.fill_diagonal(S, 1.0)
    return S


def kmo(R):
    R = np.asarray(R, float)
    inv = np.linalg.inv(R)
    d = np.sqrt(np.diag(inv))
    P = -inv / np.outer(d, d)
    np.fill_diagonal(P, 0.0)
    R0 = R.copy(); np.fill_diagonal(R0, 0.0)
    r2, p2 = R0 ** 2, P ** 2
    per_item = r2.sum(0) / (r2.sum(0) + p2.sum(0))
    overall = r2.sum() / (r2.sum() + p2.sum())
    return float(overall), per_item


def bartlett(R, n):
    R = np.asarray(R, float)
    p = R.shape[0]
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        return np.nan, np.nan, np.nan
    statistic = -(n - 1 - (2 * p + 5) / 6.0) * logdet
    df = p * (p - 1) / 2.0
    return float(statistic), float(df), float(stats.chi2.sf(statistic, df))


def paf(R, n_factors, max_iter=500, tol=1e-6):
    """Temel eksen faktörleştirme (SMC başlangıçlı, yinelemeli).

    Returns (loadings k×m, communalities, converged, heywood).
    """
    R = np.asarray(R, float)
    h2 = 1.0 - 1.0 / np.diag(np.linalg.inv(R))
    converged = False
    for _ in range(max_iter):
        Rr = R.copy(); np.fill_diagonal(Rr, h2)
        w, v = np.linalg.eigh(Rr)
        idx = np.argsort(w)[::-1][:n_factors]
        L = v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
        new = (L ** 2).sum(1)
        if np.max(np.abs(new - h2)) < tol:
            h2 = new; converged = True; break
        h2 = new
    L = L * np.sign(L.sum(0) + 1e-300)
    return L, h2, converged, bool((h2 >= 1.0).any())


def varimax(L, max_iter=500, tol=1e-8):
    p, m = L.shape
    if m < 2:
        return L.copy(), np.eye(m)
    T = np.eye(m); d = 0.0
    for _ in range(max_iter):
        Lr = L @ T
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        T = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ T, T


def promax(L, power=4):
    """Promax (Hendrickson-White). Returns (pattern, Phi)."""
    m = L.shape[1]
    if m < 2:
        return L.copy(), np.eye(m)
    V, _ = varimax(L)
    Ptarget = np.abs(V) ** power * np.sign(V)
    U = np.linalg.lstsq(V, Ptarget, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    pattern = V @ U
    Ui = np.linalg.inv(U)
    phi = Ui @ Ui.T
    return pattern, phi


def parallel_analysis(df, method, reps=50, quantile=0.95, seed=2026, factor_based=False):
    """Horn paralel analizi; marjinalleri koruyan sütun permütasyonu ve aynı
    korelasyon yöntemi kullanılır. Returns DataFrame + önerilen faktör sayısı."""
    rng = np.random.default_rng(seed)
    obs_R = correlation_matrix(df, method)["R"].to_numpy()

    def eig(R):
        if factor_based:
            R = R.copy(); np.fill_diagonal(R, 1 - 1 / np.diag(np.linalg.inv(R)))
        return np.sort(np.linalg.eigvalsh(R))[::-1]

    observed = eig(obs_R)
    x = df.to_numpy(float)
    sims = []
    for _ in range(int(reps)):
        perm = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        try:
            Rs = correlation_matrix(pd.DataFrame(perm, columns=df.columns), method)["R"].to_numpy()
        except ValueError:
            continue
        sims.append(eig(Rs))
    sims = np.array(sims)
    if len(sims) < 2:
        raise ValueError("Paralel analiz için yeterli permütasyon üretilemedi.")
    crit = np.quantile(sims, quantile, axis=0)
    above = observed > crit
    suggested = int(np.argmin(above)) if not above.all() else len(observed)
    table = pd.DataFrame({"Bileşen": np.arange(1, len(observed) + 1), "Gözlenen özdeğer": observed,
                          "Permütasyon ortalaması": sims.mean(0), f"Permütasyon %{int(quantile*100)}": crit,
                          "Korunur": above})
    return table, suggested, len(sims)


def dimensionality_analysis(df, method="auto", n_factors=None, rotation="promax",
                            pa_reps=50, seed=2026):
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] < 3:
        raise ValueError("Boyutluluk analizi için en az 3 madde gerekir.")
    cor = correlation_matrix(df, method)
    R = cor["R"].to_numpy()
    n = int(df.dropna().shape[0]) if df.notna().all(axis=1).any() else int(df.notna().sum().min())
    kmo_all, kmo_items = kmo(R)
    bart = bartlett(R, n)
    bart_valid = cor["method"] == "pearson" and cor["data_type"] == "continuous" and np.isfinite(df.to_numpy()).all() and not cor["smoothed"]
    if not bart_valid: bart = (bart[0], bart[1], np.nan)
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    pa_table, suggested, used = parallel_analysis(df, cor["method"], pa_reps, seed=seed)
    m = int(n_factors or max(suggested, 1))
    L, h2, conv, heywood = paf(R, m)
    phi = np.eye(m)
    if m > 1 and rotation == "varimax":
        L, _ = varimax(L)
    elif m > 1 and rotation == "promax":
        L, phi = promax(L)
    signs = np.sign(L.sum(0) + 1e-300)
    L = L * signs
    phi = phi * np.outer(signs, signs)
    cols = list(df.columns)
    loadings = pd.DataFrame(L, index=cols, columns=[f"F{i+1}" for i in range(m)])
    loadings["h²"] = h2
    L1, h21, conv1, _ = paf(R, 1)
    resid = R - np.outer(L1[:, 0], L1[:, 0]); np.fill_diagonal(resid, 0.0)
    iu = np.triu_indices(len(cols), 1)
    unidim = {
        "İlk özdeğer": float(eig[0]), "İkinci özdeğer": float(eig[1]),
        "Özdeğer oranı (λ1/λ2)": float(eig[0] / eig[1]),
        "İlk bileşenin açıkladığı varyans (%)": float(100 * eig[0] / len(eig)),
        "Paralel analiz önerisi": int(suggested),
        "Tek faktör artık korelasyonu |r| ort.": float(np.abs(resid[iu]).mean()),
        "Tek faktör artık korelasyonu |r| en büyük": float(np.abs(resid[iu]).max()),
        "Karar": ("Paralel analiz tek baskın boyutu destekliyor." if suggested == 1 else
                  "Paralel analiz hiçbir boyutun korunmasını desteklemiyor." if suggested == 0 else
                  f"Paralel analiz {suggested} boyut öneriyor; tek boyutluluk varsayımı desteklenmiyor."),
    }
    return {"correlation": cor, "n": n, "kmo": kmo_all,
            "kmo_items": pd.Series(kmo_items, index=cols, name="KMO (MSA)"),
            "bartlett": dict(zip(("χ²", "sd", "p"), bart)), "bartlett_inference": bool(bart_valid), "eigenvalues": eig,
            "parallel": pa_table, "pa_reps_used": used, "n_factors": m, "rotation": rotation if m > 1 else "yok",
            "loadings": loadings, "factor_correlations": pd.DataFrame(phi, index=loadings.columns[:m],
                                                                    columns=loadings.columns[:m]),
            "paf_converged": conv, "heywood": heywood,
            "one_factor_residuals": pd.DataFrame(resid, index=cols, columns=cols), "unidimensionality": unidim}
