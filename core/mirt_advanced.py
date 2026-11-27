"""Advanced MIRT production adapters (isolated; Codex-owned core modules are called, never modified).

1. Polytomous bifactor (GRM / GPCM). Python mirt 1.2.0 ``bfactor`` is binary-only and has no
   ``model="graded"`` argument or ``GradedBifactorModel``. A graded or generalized-partial-credit
   bifactor model is mathematically a confirmatory multidimensional GRM/GPCM. In it every item
   loads on the general factor and on exactly one specific factor, and the factors are orthogonal
   with unit variance. This adapter fits that model through the existing full-grid confirmatory
   engine ``core.irt_multidim.fit``. Loadings are sign-identified as positive. PCM bifactor is
   WITHHELD: fixing all slopes to one with unit factor variances is not the intended Rasch-testlet
   PCM, which needs free specific variances. That model is not supported here.
2. GRM high-precision profile. This is an explicit, user-selected EM continuation (tol 1e-11,
   max_iter 3000, item_optim_ftol 1e-14) from the default solution. It is not triggered
   automatically: no internal diagnostic has been validated as a substitute for independent R
   agreement.

All outputs are EXPERIMENTAL / NOT CERTIFIED. Item SE, confidence intervals and p-values are not produced.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

STATUS = "EXPERIMENTAL / NOT CERTIFIED"
POLY_BIFACTOR_MODELS = ("GRM", "GPCM")
WITHHELD = {
    "PCM_bifactor": "WITHHELD: unit slopes with unit variances is not the Rasch-testlet PCM (free specific variances); not validated",
    "M3PL_free": "WITHHELD in the advanced route: free c beyond the validated 2D between-item SciPy route has no accepted evidence",
    "M4PL_mixed_or_free": "WITHHELD in the advanced route: no accepted mixed/free 3D asymptote evidence",
}
HIGH_PRECISION = dict(tol=1e-11, max_iter=3000, item_optim_ftol=1e-14, item_optim_maxiter=200)


def bifactor_pattern(groups) -> np.ndarray:
    g = np.asarray(groups)
    if g.ndim != 1 or not np.issubdtype(g.dtype, np.integer) or (g < 0).any():
        raise ValueError("One non-negative integer specific-factor code per item required")
    uniq = np.unique(g)
    if len(uniq) not in (2, 3) or any(np.count_nonzero(g == k) < 3 for k in uniq):
        raise ValueError("Polytomous bifactor: 2-3 specific factors with >=3 items each (full-grid budget)")
    return np.column_stack((np.ones(len(g), int), (g[:, None] == uniq).astype(int)))


def fit_polytomous_bifactor(data, groups, model="GRM", *, n_quadpts=7, max_iter=1000, tol=1e-7, start_values=None):
    """Graded/GPCM bifactor as a confirmatory orthogonal MGRM/MGPCM on a full Gauss-Hermite grid."""
    if model not in POLY_BIFACTOR_MODELS:
        raise ValueError(WITHHELD["PCM_bifactor"] if model == "PCM" else f"Unsupported polytomous bifactor model {model!r}")
    import mirt
    from core.irt_multidim import prepare_polytomous, normalize_solver_settings
    from mirt.models.polytomous import GradedResponseModel, GeneralizedPartialCredit
    pattern = bifactor_pattern(groups); k, q = pattern.shape
    frame = pd.DataFrame(data)
    if len(frame.columns) != k or not frame.columns.is_unique or len(frame) < 30 or frame.isna().any().any():
        raise ValueError("Item count must match groups; unique names, >=30 complete rows")
    x, labels = prepare_polytomous(frame, model)
    if any(set(x[c]) != set(range(len(labels))) for c in x) or len(labels) < 3:
        raise ValueError("Every item must observe all (>=3) categories")
    if n_quadpts < 5 or n_quadpts ** q > 4000 or len(x) * k * n_quadpts ** q > 20_000_000:
        raise ValueError("Gauss-Hermite budget exceeded")
    settings = normalize_solver_settings(max_iter=max_iter, tol=tol, n_quadpts=n_quadpts)
    key = "thresholds" if model == "GRM" else "steps"
    engine = (GradedResponseModel if model == "GRM" else GeneralizedPartialCredit)(k, n_categories=len(labels), n_factors=q, item_names=list(x.columns))
    engine.set_free_parameter_masks({"discrimination": pattern.astype(bool), key: np.ones((k, len(labels) - 1), bool)})
    start = start_values if start_values is not None else {"discrimination": pattern * 1.2, key: np.tile(np.linspace(-1, 1, len(labels) - 1), (k, 1))}
    result = mirt.EMEstimator(**settings, compute_standard_errors=False, accelerate="none").fit(engine, x.to_numpy(), start=start)
    out = dict(fit=result, covariance=np.eye(q), labels=labels, model=model, settings=settings)
    a = np.asarray(out["fit"].model.parameters["discrimination"])
    if np.any(a[pattern == 0] != 0) or np.any(a[pattern == 1] <= 0):
        raise ValueError("Bifactor loading structure violated")
    if not np.allclose(out["covariance"], np.eye(pattern.shape[1]), atol=1e-12):
        raise ValueError("Bifactor factors must remain orthogonal with unit variance")
    out.update(bifactor_status=STATUS, bifactor_pattern=pattern,
               bifactor_scope="Orthogonal G+S, positive loadings, full GH grid; numerically compared with native R only on the recorded fixtures")
    return out


def _grid_probabilities(model, q, points=5):
    import itertools
    theta = np.array(list(itertools.product(np.linspace(-2, 2, points), repeat=q)))
    return np.asarray(model.probability(theta))


def fit_grm_high_precision(data, pattern, *, exploratory=False, correlated=False, n_quadpts=15, start_values=None):
    """Explicit high-precision GRM profile: default fit, then a bounded public EM continuation.

    Returns the continuation fit together with the default-versus-continuation runtime and internal
    change diagnostics. The diagnostics are descriptive. They are NOT a validated substitute for
    independent R agreement.
    """
    import mirt
    from core.irt_multidim import fit
    from mirt.models.polytomous import GradedResponseModel
    t0 = time.perf_counter()
    base = fit(data, "GRM", pattern, exploratory=exploratory, correlated=correlated, n_quadpts=n_quadpts, start_values=start_values)
    t1 = time.perf_counter()
    if correlated:
        raise ValueError("High-precision profile validated only for orthogonal GRM fits")
    x = pd.DataFrame(data).to_numpy()
    f0 = base["fit"]
    k, q = np.asarray(f0.model.parameters["discrimination"]).shape
    mask = np.asarray(f0.model.parameters["discrimination"]) != 0
    n_cat = f0.model.parameters["thresholds"].shape[1] + 1
    engine = GradedResponseModel(k, n_cat, n_factors=q, item_names=list(pd.DataFrame(data).columns))
    engine.set_free_parameter_masks(dict(discrimination=mask, thresholds=np.ones((k, n_cat - 1), bool)))
    est = mirt.EMEstimator(n_quadpts=n_quadpts, max_iter=HIGH_PRECISION["max_iter"], tol=HIGH_PRECISION["tol"],
                           item_optim_ftol=HIGH_PRECISION["item_optim_ftol"], item_optim_maxiter=HIGH_PRECISION["item_optim_maxiter"],
                           compute_standard_errors=False, accelerate="none")
    f1 = est.fit(engine, x, start=dict(f0.model.parameters))
    t2 = time.perf_counter()
    a = np.asarray(f1.model.parameters["discrimination"]); b = np.asarray(f1.model.parameters["thresholds"])
    if np.any(a[~mask] != 0) or np.any(a[mask] < .1) or np.any(a[mask] > 5):
        raise ValueError("High-precision fit violated loading constraints")
    if np.any(np.abs(b) > 6 + 1e-12) or np.any(np.diff(b, axis=1) < 1e-6 - 1e-12):
        raise ValueError("High-precision fit violated threshold constraints")
    p0, p1 = _grid_probabilities(f0.model, q), _grid_probabilities(f1.model, q)
    diagnostics = dict(default_seconds=t1 - t0, continuation_seconds=t2 - t1,
                       default_logLik=float(f0.log_likelihood), high_precision_logLik=float(f1.log_likelihood),
                       logLik_improvement=float(f1.log_likelihood - f0.log_likelihood),
                       max_parameter_change=float(max(np.max(abs(a - f0.model.parameters["discrimination"])), np.max(abs(b - f0.model.parameters["thresholds"])))),
                       max_probability_change=float(np.max(abs(p1 - p0))),
                       default_iterations=int(f0.n_iterations), continuation_iterations=int(f1.n_iterations), converged=bool(f1.converged),
                       note="Descriptive internal diagnostics only; NOT validated as an automatic trigger or as equivalent to R agreement")
    return dict(fit=f1, default=base, covariance=np.eye(q), status=STATUS, profile=dict(HIGH_PRECISION), diagnostics=diagnostics)
