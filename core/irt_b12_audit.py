"""B12.1: deterministic, computation-light audit of existing 3PL/4PL fits.

This module diagnoses stored results only; it neither refits nor asserts model
identifiability or global convergence. Safe for use on classroom-sized data.
"""
import numpy as np
import pandas as pd

_ACTIVE = {"3PL": ("a", "b", "c"), "4PL": ("a", "b", "c", "d")}


def audit_fit(fit, *, bound_tol=1e-3, likelihood_tol=1e-7):
    """Return (parameter_table, summary) for an existing EserPsychometrics fit."""
    model = fit["model"]
    if model not in _ACTIVE:
        raise ValueError("B12.1 only supports 3PL and 4PL.")
    if not (np.isfinite(bound_tol) and bound_tol > 0 and
            np.isfinite(likelihood_tol) and likelihood_tol >= 0):
        raise ValueError("Tolerances must be finite and valid.")
    items = pd.DataFrame(fit["items"])
    active = _ACTIVE[model]
    bounds = {"a": (.15, float(fit.get("a_upper", 4))),
              "b": (-4., 4.), "c": (0., .35),
              "d": (float(fit.get("d_lower", .65)), 1.)}
    rows = []
    for _, item in items.iterrows():
        for p in active:
            value = float(item[p])
            lo, hi = bounds[p]
            if not np.isfinite(value) or value < lo-bound_tol or value > hi+bound_tol:
                status = "invalid"
            elif value - lo <= bound_tol:
                status = "lower_bound"
            elif hi - value <= bound_tol:
                status = "upper_bound"
            else:
                status = "interior"
            rows.append({"item": str(item["Madde"]), "parameter": p,
                         "value": value, "position": status})
    detail = pd.DataFrame(rows)
    history = np.asarray(fit.get("history", []), dtype=float)
    history_valid = bool(history.size and np.isfinite(history).all())
    # Avoid treating tiny numerical fluctuations as substantive EM reversals.
    decreases = int(np.sum(np.diff(history) < -likelihood_tol *
                           (1 + np.abs(history[:-1])))) if history_valid else None
    convergence = fit.get("convergence") or {}
    refinement = fit.get("refinement") or {}
    bound_count = int(detail["position"].isin(("lower_bound", "upper_bound")).sum())
    invalid_count = int((detail["position"] == "invalid").sum())
    kkt = refinement.get("max_kkt_final", np.nan)
    try:
        kkt = float(kkt)
    except (TypeError, ValueError):
        kkt = np.nan
    warnings = []
    if not fit.get("converged", False):
        warnings.append("Not converged; parameters are provisional.")
    if invalid_count:
        warnings.append("Parameters outside allowable bounds or non-finite.")
    if bound_count:
        warnings.append("Active bounds; asymptotic standard errors and Wald inference require special treatment.")
    if not history_valid:
        warnings.append("Missing or invalid EM likelihood trace.")
    elif decreases:
        warnings.append("Material decrease in EM likelihood history.")
    if model == "4PL":
        warnings.append("4PL identification is not established by numerical convergence alone.")
    summary = {
        "model": model, "n": fit.get("n"), "k": len(items),
        "converged": bool(fit.get("converged", False)),
        "criterion": convergence.get("criterion", "unspecified"),
        "iterations": fit.get("iterations"),
        "bound_parameters": bound_count, "invalid_parameters": invalid_count,
        "history_valid": history_valid, "likelihood_decreases": decreases,
        "kkt_per_person": kkt if np.isfinite(kkt) else None,
        "inference_ready": bool(fit.get("converged", False) and
                                not bound_count and not invalid_count and
                                history_valid and decreases == 0),
        "warnings": warnings,
    }
    # A necessary safety gate, never a sufficient proof of valid inference.
    return detail, summary
