"""Independent known-truth dichotomous IRT simulation and parameter recovery.

The data generator uses NumPy logistic probabilities, NOT mirt.simdata.
Metrics are reported per parameter, without assuming convergence or equivalence.
Run: PYTHONPATH=. python validation/known_truth_irt_recovery.py
"""
import json
import numpy as np
import mirt
from scipy.special import expit
from core.irt_python_mirt_parameters import python_mirt_binary_parameters

MODELS = ("1PL", "2PL", "3PL", "4PL")


def truth(model, n_items):
    b = np.linspace(-1.25, 1.25, n_items)
    a = np.ones(n_items) if model == "1PL" else np.linspace(0.8, 1.6, n_items)
    c = np.linspace(0.10, 0.18, n_items) if model in ("3PL", "4PL") else np.zeros(n_items)
    d = np.linspace(0.90, 0.97, n_items) if model == "4PL" else np.ones(n_items)
    return {"a": a, "b": b, "c": c, "d": d}


def simulate(model, *, n_persons=1000, n_items=12, seed=2026):
    if model not in MODELS:
        raise ValueError("Unknown model")
    if n_persons < 2 or n_items < 2:
        raise ValueError("At least two persons and items required")
    rng = np.random.default_rng(seed)
    params = truth(model, n_items)
    theta = rng.standard_normal(n_persons)
    p = params["c"][None, :] + (
        params["d"] - params["c"])[None, :] * expit(
        params["a"][None, :] * (theta[:, None] - params["b"][None, :]))
    x = (rng.random(p.shape) < p).astype(int)
    return x, params


def evaluate(model, *, n_persons=1000, n_items=12, seed=2026):
    x, actual = simulate(model, n_persons=n_persons, n_items=n_items, seed=seed)
    result = {"model": model, "seed": seed, "n_persons": n_persons,
              "n_items": n_items, "status": "FAILED", "metrics": {},
              "converged": None}
    try:
        fit = mirt.fit_mirt(x, model=model)
        table = python_mirt_binary_parameters(fit, model=model)
        convergence = getattr(fit, "converged", None)
        if callable(convergence):
            convergence = convergence()
        result["converged"] = bool(convergence) if convergence is not None else None
        parameters = ("a", "b") + (("c",) if model in ("3PL", "4PL") else ()) + (
            ("d",) if model == "4PL" else ())
        for param in parameters:
            estimated = table[param].to_numpy(dtype=float)
            true = actual[param]
            if estimated.shape != true.shape or not np.isfinite(estimated).all():
                raise ValueError("Nonfinite or incompatible coefficient table")
            residual = estimated - true
            result["metrics"][param] = {
                "bias": float(residual.mean()),
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "max_absolute_error": float(np.abs(residual).max()),
            }
        result["status"] = "MEASURED" if result["converged"] is not False else "NONCONVERGED"
    except (ValueError, TypeError, RuntimeError, AttributeError, ArithmeticError) as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:300]
    return result


def run():
    records = [evaluate(model, seed=seed) for model in MODELS
               for seed in (2026, 2027)]
    return {
        "generator": "Independent NumPy logistic 1PL-4PL with known a,b,c,d",
        "theta_distribution": "standard normal",
        "engine": "Python mirt 1.2.0",
        "notes": [
            "Bias, MAE, RMSE are empirical; no pass/fail threshold is imposed.",
            "3PL/4PL recovery is sensitive to identification, sample size and optimization.",
            "Not an independent software-engine comparison or R mirt equivalence test.",
        ],
        "runs": records,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
