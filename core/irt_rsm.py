"""Rating Scale Model (Andrich, 1978) — complete single-group MML analysis chain.

Model (ordered categories c = 0..C-1, common steps for every item):
    P(X_i = c | theta) = exp(sum_{s<=c} (theta - delta_i - tau_s)) / sum_h exp(sum_{s<=h} (theta - delta_i - tau_s)),
    with the empty sum (c = 0) equal to 0, sum_s tau_s = 0 (C - 1 steps, C - 2 free), slope fixed at 1.
Identification ("rasch", default): theta ~ N(0, sigma^2), latent mean 0, sigma estimated (Rasch family).
Identification ("unit_variance"): theta ~ N(0, 1); same item/step parameterization (for comparison with
    libraries that fix the latent variance, e.g. Python mirt RatingScaleModel, which fixes its first
    threshold at 0 instead of sum tau = 0; see `to_mirt_parameterization`).
The partial credit model (PCM, item-specific steps b_is) is implemented ONLY as the nested alternative
for the RSM-vs-PCM likelihood-ratio test under the same latent specification; it is labeled PCM.
This module is independent of core.polytomous_dif (two-group fixed-anchor RSM DIF, sigma = 1).

Estimation: marginal ML on Gauss-Hermite nodes (theta = sigma * z), analytic score, L-BFGS-B.
Convergence: max |score| per person <= core.rasch.SCORE_TOLERANCE and no parameter at a box bound.
SEs: inverse observed information (central differences of the analytic score, symmetrized); withheld
unless converged and positive definite. Missing responses are skipped (MAR).
Persons: EAP (posterior SD) and Warm WLE (finite for extreme scores) with model SE 1/sqrt(I).
Item fit: infit/outfit MNSQ and ZSTD (WLE, extreme persons excluded). Information I_i = Var(X_i | theta).
"""
import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2

from core.rasch import SCORE_TOLERANCE

LOC_BOUND = 8.0
STEP_BOUND = 6.0
LOG_SD_BOUNDS = (np.log(0.05), np.log(10.0))
IDENTIFICATIONS = ("rasch", "unit_variance")


# ------------------------------------------------------------------ data
def prepare(data, min_persons=30, min_items=2):
    frame = pd.DataFrame(data)
    names = [str(c) for c in frame.columns]
    x = frame.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bad = frame.notna().to_numpy() & np.isnan(x)
    if bad.any():
        raise ValueError("Sayısal olmayan yanıt var; RSM 0,1,2,... sıralı puan gerektirir.")
    obs = np.isfinite(x)
    if obs.any() and (not np.array_equal(x[obs], np.floor(x[obs])) or (x[obs] < 0).any()):
        raise ValueError("Yanıtlar 0'dan başlayan tam sayı kategori puanları olmalıdır.")
    keep = obs.any(axis=1)
    x, obs = x[keep], obs[keep]
    n, k = x.shape
    if n < min_persons or k < min_items:
        raise ValueError(f"En az {min_persons} kişi ve {min_items} madde gereklidir (N={n}, k={k}).")
    C = int(np.nanmax(x)) + 1
    if C < 3:
        raise ValueError("RSM en az 3 sıralı kategori gerektirir; ikili veriler için Rasch kullanın.")
    xi = np.where(obs, x, 0).astype(int)
    counts = np.array([np.bincount(xi[obs[:, i], i], minlength=C) for i in range(k)])
    if (counts.sum(0) == 0).any():
        missing = np.flatnonzero(counts.sum(0) == 0).tolist()
        raise ValueError(f"Hiç gözlenmeyen kategori(ler): {missing}. Ortak adım parametresi tanımsız; "
                         "kategoriler sessizce birleştirilmez.")
    constant = [names[i] for i in range(k) if (counts[i] > 0).sum() < 2]
    if constant:
        raise ValueError("Yanıt değişkenliği olmayan madde(ler): " + ", ".join(constant))
    return {"x": xi, "obs": obs, "keep": keep, "names": names, "C": C, "counts": counts,
            "missing": int((~obs).sum())}


# ------------------------------------------------------------------ model algebra
def category_log_probs(theta, steps):
    """theta (Q,), steps (k, C-1) -> log P (Q, k, C)."""
    theta = np.asarray(theta, float)
    k, m = steps.shape
    cum = np.concatenate([np.zeros((k, 1)), np.cumsum(steps, axis=1)], axis=1)        # (k, C)
    eta = theta[:, None, None] * np.arange(m + 1)[None, None, :] - cum[None, :, :]
    return eta - logsumexp(eta, axis=2, keepdims=True)


def rsm_steps(delta, tau):
    return np.asarray(delta, float)[:, None] + np.asarray(tau, float)[None, :]


def item_moments(theta, steps):
    P = np.exp(category_log_probs(theta, steps))
    c = np.arange(P.shape[2])
    E = (P * c).sum(2); V = (P * c ** 2).sum(2) - E ** 2
    K3 = (P * (c - E[..., None]) ** 3).sum(2); K4 = (P * (c - E[..., None]) ** 4).sum(2)
    return P, E, V, K3, K4


def information(theta, steps):
    """Rasch-family item information I_i(theta) = Var(X_i | theta) (Q, k)."""
    return item_moments(theta, steps)[2]


class _Spec:
    def __init__(self, model, k, C, identification):
        if model not in ("RSM", "PCM"):
            raise ValueError("Model RSM veya PCM olmalıdır.")
        if identification not in IDENTIFICATIONS:
            raise ValueError("Tanımlama 'rasch' veya 'unit_variance' olmalıdır.")
        self.model, self.k, self.C, self.ident = model, k, C, identification
        self.n_item = k if model == "RSM" else k * (C - 1)
        self.n_tau = C - 2 if model == "RSM" else 0
        self.free_sd = identification == "rasch"
        self.size = self.n_item + self.n_tau + int(self.free_sd)

    def unpack(self, v):
        if self.model == "RSM":
            delta = v[:self.k]; tf = v[self.k:self.k + self.n_tau]
            tau = np.r_[tf, -tf.sum()]
            steps = rsm_steps(delta, tau)
        else:
            delta = tau = None
            steps = v[:self.n_item].reshape(self.k, self.C - 1)
        sd = float(np.exp(v[-1])) if self.free_sd else 1.0
        return steps, sd, delta, tau

    def bounds(self):
        b = ([(-LOC_BOUND, LOC_BOUND)] * self.k + [(-STEP_BOUND, STEP_BOUND)] * self.n_tau if self.model == "RSM"
             else [(-LOC_BOUND - STEP_BOUND, LOC_BOUND + STEP_BOUND)] * self.n_item)
        return b + ([LOG_SD_BOUNDS] if self.free_sd else [])

    def chain(self, G):
        """Gradient w.r.t. steps (k, C-1) -> gradient w.r.t. item/step parameters."""
        if self.model == "PCM":
            return G.reshape(-1)
        g_delta = G.sum(1)
        g_tau_full = G.sum(0)
        g_tau = g_tau_full[:-1] - g_tau_full[-1]
        return np.r_[g_delta, g_tau]


def _nodes(points):
    z, w = hermegauss(points)
    return z, np.log(w / w.sum())


def _negloglik(v, spec, onehot, obs, ge_counts, sumx, z, lw, n_total):
    steps, sd, _, _ = spec.unpack(v)
    theta = sd * z
    logP = category_log_probs(theta, steps)                                  # (Q, k, C)
    ll = np.einsum("nkc,qkc->nq", onehot, logP) + lw[None, :]
    norm_ = logsumexp(ll, axis=1)
    post = np.exp(ll - norm_[:, None])                                        # (n, Q)
    r = post.T @ obs                                                          # (Q, k) expected responders
    P = np.exp(logP)
    S = np.flip(np.cumsum(np.flip(P, 2), 2), 2)[:, :, 1:]                     # P(X >= s), s = 1..C-1
    G = -ge_counts + np.einsum("qk,qks->ks", r, S)                            # d loglik / d steps
    grad = spec.chain(G)
    if spec.free_sd:
        E = (P * np.arange(spec.C)).sum(2)
        A = post.T @ sumx
        g_ls = float((theta * A).sum() - (theta[:, None] * r * E).sum())
        grad = np.r_[grad, g_ls]
    return -norm_.sum() / n_total, -grad / n_total


def _start(prep, spec):
    x, obs, C = prep["x"], prep["obs"], prep["C"]
    mean = np.array([x[obs[:, i], i].mean() for i in range(spec.k)])
    loc = np.clip(-np.log(np.clip(mean / (C - 1), .02, .98) / (1 - np.clip(mean / (C - 1), .02, .98))), -3, 3)
    tot = prep["counts"].sum(0).astype(float) + .5
    tau0 = -np.diff(np.log(tot)); tau0 = np.clip(tau0 - tau0.mean(), -3, 3)
    if spec.model == "RSM":
        v = np.r_[loc, tau0[:-1]]
    else:
        v = (loc[:, None] + tau0[None, :]).reshape(-1)
    return np.r_[v, [0.] if spec.free_sd else []]


def _observed_information(v, args, spec):
    m = len(v); H = np.zeros((m, m)); n = args[-1]
    for j in range(m):
        h = 1e-5 * max(1., abs(v[j]))
        up, dn = v.copy(), v.copy(); up[j] += h; dn[j] -= h
        H[:, j] = (_negloglik(up, spec, *args)[1] - _negloglik(dn, spec, *args)[1]) * n / (2 * h)
    return (H + H.T) / 2


def fit(data, model="RSM", identification="rasch", points=41, max_iter=5000):
    """Marginal ML fit; returns parameters, SEs (when valid), fit indices and diagnostics."""
    prep = prepare(data)
    x, obs, C = prep["x"], prep["obs"], prep["C"]
    n, k = x.shape
    spec = _Spec(model, k, C, identification)
    onehot = (np.arange(C)[None, None, :] == x[:, :, None]) & obs[:, :, None]
    onehot = onehot.astype(float)
    ge_counts = np.stack([((x >= s) & obs).sum(0) for s in range(1, C)], axis=1).astype(float)
    sumx = np.where(obs, x, 0).sum(1).astype(float)
    z, lw = _nodes(points)
    args = (onehot, obs.astype(float), ge_counts, sumx, z, lw, n)
    bounds = spec.bounds(); lo, hi = np.array(bounds).T
    res = minimize(_negloglik, _start(prep, spec), args=(spec,) + args, jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 1e-12, "maxcor": 30})
    v = np.asarray(res.x, float)
    f, g = _negloglik(v, spec, *args)
    at_bound = (v - lo <= 1e-6) | (hi - v <= 1e-6)
    score = float(np.abs(g[~at_bound]).max()) if (~at_bound).any() else 0.
    converged = bool(np.isfinite(f) and score <= SCORE_TOLERANCE and not at_bound.any())
    loglik = -f * n
    steps, sd, delta, tau = spec.unpack(v)
    cov, se_reason = None, ""
    if not converged:
        se_reason = "Kestirim yakınsamadı veya bir parametre sınırda; SH raporlanmaz."
    else:
        info = _observed_information(v, args, spec)
        eig = np.linalg.eigvalsh(info)
        if not np.isfinite(eig).all() or eig[0] <= 1e-8 * max(eig[-1], 1.):
            se_reason = "Gözlenen bilgi matrisi pozitif tanımlı değil; SH raporlanmaz."
        else:
            cov = np.linalg.inv(info)
    npar = spec.size
    out = {"model": model, "identification": identification, "n": n, "k": k, "C": C, "names": prep["names"],
           "params": v, "spec": spec, "steps": steps, "sd": sd, "loglik": float(loglik),
           "aic": float(2 * npar - 2 * loglik), "bic": float(np.log(n) * npar - 2 * loglik), "npar": npar,
           "converged": converged, "max_score_per_person": score, "iterations": int(res.nit),
           "optimizer_message": str(res.message), "at_bound": bool(at_bound.any()), "cov": cov,
           "se_reason": se_reason, "points": int(points), "x": x, "obs": obs, "kept_rows": prep["keep"],
           "missing": prep["missing"], "counts": prep["counts"]}
    if model == "RSM":
        out["delta"], out["tau"] = delta, tau
    out["tables"] = _tables(out)
    return out


def _tables(f):
    spec, cov, names = f["spec"], f["cov"], f["names"]
    se = np.sqrt(np.diag(cov)) if cov is not None else np.full(spec.size, np.nan)
    z = 1.959963984540054
    if f["model"] == "RSM":
        items = pd.DataFrame({"Madde": names, "Konum (δ)": f["delta"], "SH(δ)": se[:spec.k]})
        items["δ %95 alt"] = items["Konum (δ)"] - z * items["SH(δ)"]
        items["δ %95 üst"] = items["Konum (δ)"] + z * items["SH(δ)"]
        # last step is derived: tau_{C-1} = -sum(free); var via the gradient (-1, ..., -1)
        tse = np.full(f["C"] - 1, np.nan)
        if cov is not None:
            sub = cov[spec.k:spec.k + spec.n_tau, spec.k:spec.k + spec.n_tau]
            tse[:-1] = np.sqrt(np.diag(sub)); tse[-1] = np.sqrt(sub.sum())
        steps = pd.DataFrame({"Adım": [f"τ{s}" for s in range(1, f["C"])], "Ortak adım (τ)": f["tau"], "SH(τ)": tse})
        steps["τ %95 alt"] = steps["Ortak adım (τ)"] - z * steps["SH(τ)"]
        steps["τ %95 üst"] = steps["Ortak adım (τ)"] + z * steps["SH(τ)"]
        steps["Not"] = ["serbest"] * (f["C"] - 2) + ["türetilmiş: −Σ diğerleri"]
    else:
        st = f["steps"]; sst = se[:spec.n_item].reshape(spec.k, f["C"] - 1)
        items = pd.DataFrame({"Madde": names})
        for s in range(f["C"] - 1):
            items[f"b{s + 1}"] = st[:, s]; items[f"SH(b{s + 1})"] = sst[:, s]
        steps = None
    sd_se = float(f["sd"] * se[-1]) if spec.free_sd else np.nan
    latent = pd.DataFrame([{"Örtük ortalama": 0.0, "Örtük SS (σ)": f["sd"], "SH(σ)": sd_se,
                            "Tanımlama": "σ serbest (Rasch)" if spec.free_sd else "σ = 1 sabit"}])
    return {"items": items, "steps": steps, "latent": latent}


# ------------------------------------------------------------------ persons
def eap(f, points=61):
    z, lw = _nodes(points); theta = f["sd"] * z
    logP = category_log_probs(theta, f["steps"])
    onehot = ((np.arange(f["C"])[None, None, :] == f["x"][:, :, None]) & f["obs"][:, :, None]).astype(float)
    ll = np.einsum("nkc,qkc->nq", onehot, logP) + lw[None, :]
    post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    m = post @ theta
    return m, np.sqrt(np.maximum(post @ theta ** 2 - m ** 2, 0.))


def wle(f, lo=-10., hi=10., iterations=100):
    """Warm WLE: sum(x - E) + J/(2I) = 0 with I = sum Var, J = sum third central moment."""
    x, obs, steps = f["x"], f["obs"], f["steps"]
    n = len(x); a = np.full(n, lo); b = np.full(n, hi)
    def eq(t):
        out = np.empty(n)
        for chunk in np.array_split(np.arange(n), max(1, n // 2000)):
            _, E, V, K3, _ = item_moments(t[chunk], steps)                    # (m, k)
            o = obs[chunk]
            I = np.where(o, V, 0).sum(1); J = np.where(o, K3, 0).sum(1)
            out[chunk] = np.where(o, x[chunk] - E, 0).sum(1) + J / (2 * np.maximum(I, 1e-12))
        return out
    fa = eq(a)
    for _ in range(iterations):
        mid = (a + b) / 2; fm = eq(mid)
        left = np.sign(fm) == np.sign(fa)
        a = np.where(left, mid, a); fa = np.where(left, fm, fa); b = np.where(left, b, mid)
    theta = (a + b) / 2
    _, _, V, _, _ = item_moments(theta, steps)
    I = np.where(obs, V, 0).sum(1)
    return theta, 1 / np.sqrt(np.maximum(I, 1e-12))


def persons(f):
    th_w, se_w = wle(f); th_e, sd_e = eap(f)
    raw = np.where(f["obs"], f["x"], 0).sum(1); mx = f["obs"].sum(1) * (f["C"] - 1)
    return pd.DataFrame({"Ham puan": raw.astype(int), "Azami puan": mx.astype(int), "WLE θ": th_w, "SH(θ) WLE": se_w,
                         "EAP θ": th_e, "Posterior SS": sd_e, "Uç puan": (raw == 0) | (raw == mx)})


# ------------------------------------------------------------------ fit statistics
def _zstd(ms, q):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.cbrt(ms) - 1) * (3 / q) + q / 3


def item_fit(f, theta=None):
    x, obs, steps = f["x"], f["obs"], f["steps"]
    if theta is None:
        theta, _ = wle(f)
    raw = np.where(obs, x, 0).sum(1); mx = obs.sum(1) * (f["C"] - 1)
    use = (raw > 0) & (raw < mx)
    _, E, V, _, K4 = item_moments(theta[use], steps)
    o = obs[use]; y = x[use]
    R2 = np.where(o, (y - E) ** 2, 0); W = np.where(o, V, 0); Kc = np.where(o, K4, 0)
    nobs = o.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = np.where(o, R2 / np.maximum(W, 1e-12), 0)
        outfit = z2.sum(0) / nobs
        infit = R2.sum(0) / W.sum(0)
        q_out = np.sqrt(np.maximum((np.where(o, Kc / np.maximum(W, 1e-12) ** 2, 0)).sum(0) / nobs ** 2 - 1 / nobs, 1e-12))
        q_in = np.sqrt(np.maximum((Kc - W ** 2).sum(0) / W.sum(0) ** 2, 1e-12))
    return pd.DataFrame({"Madde": f["names"], "Infit MNSQ": infit, "Infit ZSTD": _zstd(infit, q_in),
                         "Outfit MNSQ": outfit, "Outfit ZSTD": _zstd(outfit, q_out), "Kullanılan kişi": nobs})


def rsm_vs_pcm(data, identification="rasch", points=41):
    """Nested LR test: RSM (common steps) vs PCM (item-specific steps), same latent specification."""
    r = fit(data, "RSM", identification, points); p = fit(data, "PCM", identification, points)
    df = p["npar"] - r["npar"]
    ok = r["converged"] and p["converged"] and p["loglik"] >= r["loglik"] - 1e-6
    stat = max(2 * (p["loglik"] - r["loglik"]), 0.)
    return {"LR": stat, "df": df, "p": float(chi2.sf(stat, df)) if ok else np.nan,
            "valid": ok, "rsm": r, "pcm": p,
            "note": "" if ok else "Modellerden biri yakınsamadı veya iç içe olabilirlik ihlali; p değeri verilmedi.",
            "table": pd.DataFrame([{"Model": "RSM", "logLik": r["loglik"], "Parametre": r["npar"], "AIC": r["aic"],
                                    "BIC": r["bic"], "Yakınsama": r["converged"]},
                                   {"Model": "PCM", "logLik": p["loglik"], "Parametre": p["npar"], "AIC": p["aic"],
                                    "BIC": p["bic"], "Yakınsama": p["converged"]}])}


def category_curves(f, theta=None):
    theta = np.linspace(-4, 4, 161) if theta is None else np.asarray(theta, float)
    P = np.exp(category_log_probs(theta, f["steps"]))
    return theta, P


def to_mirt_parameterization(delta, tau):
    """Map (delta, centered tau) to Python mirt RatingScaleModel (thresholds[0] = 0)."""
    tau = np.asarray(tau, float); delta = np.asarray(delta, float)
    return {"difficulty": delta + tau[0], "thresholds": np.r_[0., tau[1:] - tau[0]]}


def simulate(delta, tau, n, sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    th = sd * rng.standard_normal(n)
    P = np.exp(category_log_probs(th, rsm_steps(delta, tau)))
    u = rng.random((n, len(delta)))[..., None]
    return (u > np.cumsum(P, 2)).sum(2)
