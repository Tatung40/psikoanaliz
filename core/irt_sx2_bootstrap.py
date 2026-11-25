"""Reference-scoped fitted-null parametric bootstrap p-values for the UNCHANGED production S-X2 (P3 closure J1).

B=99 bootstrap datasets from the fitted 3PL/4PL (theta~N(0,1), Bernoulli), each with a full cold native MAP refit and
full S-X2 recomputation including the library's raw-score regrouping; p = (1 + #{S* >= S}) / (B + 1). More than one
failed refit, or a non-finite statistic, makes the item non-evaluable (no p-value). Same algorithm as
validation/p3_sx2_bootstrap/run.py (identity-tested). Chi-square p-values stay closed.
"""
import numpy as np
from scipy.special import expit

B = 99
METHOD = "S-X² parametrik bootstrap p (uydurulmuş null, B=99, her örnekte tam soğuk MAP yeniden kestirimi ve yeniden gruplama)"
COLS = ("discrimination", "difficulty", "guessing", "upper")


def bootstrap_pvalues(outcome, responses, seed=(20261125,), n_boot=B):
    """(p-values per item or None, reason). Caller must check reference scope first."""
    import mirt
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_binary_onestep import onestep_scope
    if not onestep_scope(outcome):
        return None, "Bootstrap S-X² p yalnız doğrulanmış referans kapsamında (N=3000, 12 madde, 3PL/4PL MAP) üretilir."
    x = np.asarray(responses).astype(np.uint8); model = outcome.model; k = 4 if model == "4PL" else 3
    s = mirt.itemfit(outcome.fit, x, statistics=["S_X2"])["S_X2"].to_numpy(float)
    e = outcome.fit.coef()[list(COLS[:k])].to_numpy(float)
    a, b, c = e[:, 0], e[:, 1], e[:, 2]; d = e[:, 3] if k == 4 else np.ones(e.shape[0])
    star = np.full((n_boot, e.shape[0]), np.nan); fails = 0
    for i in range(n_boot):
        rng = np.random.default_rng([*seed, i + 1, 0]); th = rng.standard_normal(x.shape[0])
        xs = (rng.random(x.shape) < c + (d - c) * expit(a * (th[:, None] - b))).astype(np.uint8)
        try:
            f = fit_python_mirt(xs, model=model, estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit
            if not f.converged: fails += 1; continue
            star[i] = mirt.itemfit(f, xs, statistics=["S_X2"])["S_X2"].to_numpy(float)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            fails += 1
    if fails > 1:
        return None, f"{fails}/{n_boot} bootstrap yeniden kestirimi başarısız; p üretilmedi (reddedildi)."
    p = (1 + np.where(np.isfinite(star), star >= s, True).sum(0)) / (n_boot + 1)
    return np.where(np.isfinite(s), p, np.nan), None
