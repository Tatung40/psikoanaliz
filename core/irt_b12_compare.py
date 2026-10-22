"""B12.2: reproducible side-by-side comparisons of existing estimators.

Never interprets a larger likelihood as evidence of parameter recovery.
Runs are kept separate: single EM, 3-start EM, and EM plus refinement.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from core.irt import fit_irt
from core.irt_multistart import fit_irt_multistart
from core.irt_b12_audit import audit_fit


def compare_estimators(data, model="4PL", *, points=31, max_iter=150,
                       a_upper=4., d_lower=.65, starts=(0., -1., 1.)):
    if model not in ("3PL", "4PL"):
        raise ValueError("Only 3PL and 4PL supported.")
    if not starts or not np.isfinite(starts).all():
        raise ValueError("Provide valid starting shifts.")
    base = dict(model=model, points=points, max_iter=max_iter,
                a_upper=a_upper, d_lower=d_lower)
    settings = (
        ("EM", lambda: fit_irt(data, **base)),
        ("Multistart EM", lambda: fit_irt_multistart(
            data, starts=starts, **base)["fit"]),
        ("EM + L-BFGS-B", lambda: fit_irt(data, refine=True, **base)),
    )
    rows, fits = [], {}
    for label, run in settings:
        t0 = perf_counter()
        try:
            fit = run()
            _, audit = audit_fit(fit)
            fits[label] = fit
            rows.append(dict(method=label, loglik=float(fit["loglik"]),
                             converged=bool(fit["converged"]),
                             iterations=int(fit["iterations"]),
                             bound_parameters=audit["bound_parameters"],
                             invalid_parameters=audit["invalid_parameters"],
                             likelihood_decreases=audit["likelihood_decreases"],
                             kkt_per_person=audit["kkt_per_person"],
                             status="completed", error="",
                             seconds=perf_counter()-t0))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append(dict(method=label, loglik=np.nan, converged=False,
                             iterations=np.nan, bound_parameters=np.nan,
                             invalid_parameters=np.nan, likelihood_decreases=np.nan,
                             kkt_per_person=np.nan, status="failed",
                             error=str(exc), seconds=perf_counter()-t0))
    result = pd.DataFrame(rows)
    return result, fits
