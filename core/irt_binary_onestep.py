"""Reference-scoped frequentist item-parameter CI for native 3PL/4PL MAP (P1 closure J).

The displayed MAP estimates are unchanged. A separate interval estimator is reported: one Newton step from the MAP
estimate to the likelihood equations, eta_os = eta_MAP - I(eta_MAP)^-1 grad log prior / N, which removes the first-order
prior-induced bias, with Wald interval eta_os +/- 1.96 sqrt(diag I^-1 / N) intersected with the native parameter box.
I is the exact expected marginal Fisher information over all 2^12 response patterns (61-node quadrature of the fit).
SE column = SE of eta_os, never of the MAP estimate. Independent confirmation: validation/p3_sx2_analytic/PROTOCOL.json,
validation/evidence/p1p3_analytic/REPORT_J.json (3000 fresh calibrations/model, N=3000, J=12 reference design).
"""
import numpy as np
import pandas as pd
from scipy.special import expit

J = 12
METHOD = "Tek adımlı önsel-yanlılık düzeltmeli Wald GA (beklenen Fisher bilgisi, 4096 örüntü)"
SCOPE = ("Yalnız N=3000, 12 madde, Python mirt 1.2.0 MAP (61 düğüm, tol 1e-4, max_iter 500), kayıtlı madde tasarımı "
         "(validation.binary_certification.common.truth) ve θ~N(0,1) referans tasarımında bağımsız olarak doğrulandı. Uygulama "
         "kullanıcı verisinin madde tasarımını ve örtük dağılımını doğrulayamaz; bu satırlar o veri için sertifika kanıtı değildir.")
# Audit (validation/evidence/p1p3_analytic/AUDIT_closureJ.json): 4PL c/d intervals are whole-box in 46%/39% of
# replications and cover only .9005/.9141 when informative -> CI certificates withdrawn; their SE certificates remain.
CI_VALIDATED = {"3PL": {"discrimination", "difficulty", "guessing"}, "4PL": {"discrimination", "difficulty"}}
SE_VALIDATED = {"3PL": {"discrimination", "difficulty", "guessing"}, "4PL": {"discrimination", "difficulty", "guessing", "upper"}}
WITHDRAWN = ("NOT CERTIFIED — 4PL c/d GA: aralıkların %46/%39'u tüm parametre kutusu, bilgi veren aralıklarda kapsama "
             ".9005/.9141 (<.93); Wald GA üretilmez. Tek-adım SH doğrulandı; doğrulanmış C(α) güven kümesi ayrıca sunulur.")
NAMES = ("discrimination", "difficulty", "guessing", "upper")
BOX = {"discrimination": (0.0, np.inf), "difficulty": (-np.inf, np.inf), "guessing": (0.0, .5), "upper": (.5, 1.0)}


def _patterns():
    return ((np.arange(2 ** J)[:, None] >> np.arange(J)) & 1).astype(float)


def _quadrature():
    from mirt.scoring._common import build_quadrature
    t, w = build_quadrature(n_quadpts=61, n_factors=1, prior_mean=None, prior_cov=None)
    return np.asarray(t, float).ravel(), np.asarray(w, float).ravel()


def expected_information(eta, model):
    """Per-person expected marginal Fisher information (exact pattern enumeration), physical (a,b,c[,d]) order."""
    k = 4 if model == "4PL" else 3; e = np.asarray(eta, float).reshape(J, k)
    a, b, c = e[:, 0], e[:, 1], e[:, 2]; d = e[:, 3] if k == 4 else np.ones(J)
    t, w = _quadrature(); X = _patterns()
    z = expit(a * (t[:, None] - b)); P = np.clip(c + (d - c) * z, 1e-300, 1 - 1e-16); dz = z * (1 - z)
    g = [(d - c) * dz * (t[:, None] - b), -(d - c) * dz * a, 1 - z] + ([z] if k == 4 else [])
    dP = np.stack(g, 2)
    ll = X @ np.log(P).T + (1 - X) @ np.log1p(-P).T; m = ll.max(1, keepdims=True); L = np.exp(ll - m) * w
    px = L.sum(1); post = L / px[:, None]; px = px * np.exp(m[:, 0])
    score = np.einsum("xq,xqj,qjk->xjk", post, (X[:, None, :] - P[None]) / (P * (1 - P))[None], dP).reshape(len(X), -1)
    return (px[:, None] * score).T @ score


def log_prior_gradient(eta, model):
    """Gradient of the library default MAP log prior: a~LogN(0,.5), b~N(0,2), c~Beta(2,8), d~Beta(8,2)."""
    k = 4 if model == "4PL" else 3; e = np.asarray(eta, float).reshape(J, k)
    a, b, c = e[:, 0], e[:, 1], e[:, 2]
    g = [-1 / a - np.log(a) / (.25 * a), -b / 4, 1 / c - 7 / (1 - c)]
    if k == 4: d = e[:, 3]; g.append(7 / d - 1 / (1 - d))
    return np.stack(g, 1).ravel()


def onestep_scope(outcome):
    """Eligibility for the independently confirmed reference scope (never a proof of population assumptions)."""
    if getattr(outcome, "model", None) not in ("3PL", "4PL") or getattr(outcome, "estimation", None) != "MAP":
        return False
    if (getattr(outcome, "n_persons", None), getattr(outcome, "n_items", None)) != (3000, J):
        return False
    if getattr(outcome, "solver_settings", {}) != dict(max_iter=500, tol=1e-4, n_quadpts=61):
        return False
    import importlib.metadata
    return importlib.metadata.version("mirt") == "1.2.0" and bool(getattr(outcome.fit, "converged", False))


def onestep_intervals(outcome, item_names):
    """(table, reason). table is None outside scope or when the information matrix is refused."""
    if not onestep_scope(outcome):
        return None, "Tek adımlı frekansçı GA yalnız doğrulanmış referans kapsamında (N=3000, 12 madde, 3PL/4PL MAP) üretilir."
    model = outcome.model; k = 4 if model == "4PL" else 3; names = NAMES[:k]
    coef = outcome.fit.coef()
    eta = coef[list(names)].to_numpy(float).ravel()
    I = expected_information(eta, model); ev = np.linalg.eigvalsh((I + I.T) / 2)
    if not np.all(np.isfinite(I)) or ev[0] <= 0 or ev[-1] / ev[0] > 1e12:
        return None, "Beklenen bilgi matrisi pozitif tanımlı değil veya koşul sayısı > 1e12; GA üretilmedi (reddedildi)."
    N = outcome.n_persons; Ii = np.linalg.inv(I)
    center = eta - Ii @ log_prior_gradient(eta, model) / N; se = np.sqrt(np.diag(Ii) / N)
    rows = []
    for j, item in enumerate(item_names):
        for q, name in enumerate(names):
            i = j * k + q; lo, hi = BOX[name]; ci_ok = name in CI_VALIDATED[model]
            rows.append({"Madde": str(item), "Parametre": name, "MAP kestirimi": eta[i], "Tek adım merkezi": center[i],
                         "SH (tek adım)": se[i] if name in SE_VALIDATED[model] else np.nan,
                         "%95 GA alt": max(center[i] - 1.959964 * se[i], lo) if ci_ok else np.nan,
                         "%95 GA üst": min(center[i] + 1.959964 * se[i], hi) if ci_ok else np.nan, "Yöntem": METHOD,
                         "Bilimsel durum": ("REFERANS KAPSAMINDA DOĞRULANDI — frekansçı GA ve tek-adım SH; MAP SH'si değildir"
                                            if ci_ok else WITHDRAWN), "Kapsam": SCOPE})
    return pd.DataFrame(rows), None
