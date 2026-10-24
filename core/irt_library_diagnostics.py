"""Read-only display diagnostics from Python mirt FitResult.

No native engine inference or invented convergence metrics.
"""
import math


def library_fit_diagnostics(fit):
    required = ("converged", "n_iterations", "log_likelihood",
                "aic", "bic", "n_parameters", "n_observations")
    missing = [key for key in required if not hasattr(fit, key)]
    if missing:
        raise ValueError("Python mirt result missing fields: " + ", ".join(missing))
    result = {
        "converged": bool(fit.converged),
        "iterations": int(fit.n_iterations),
        "log_likelihood": float(fit.log_likelihood),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "n_parameters": int(fit.n_parameters),
        "n_observations": int(fit.n_observations),
    }
    if result["iterations"] < 0 or result["n_parameters"] < 1 or result["n_observations"] < 1:
        raise ValueError("Invalid Python mirt diagnostic counts")
    for key in ("log_likelihood", "aic", "bic"):
        if not math.isfinite(result[key]):
            raise ValueError("Nonfinite Python mirt diagnostic: " + key)
    return result
