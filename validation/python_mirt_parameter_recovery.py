"""Simulation-based parameter recovery for Python mirt, not cross-engine certification.

Run with: python validation/python_mirt_parameter_recovery.py
Outputs machine-readable JSON; fixed seeds, declared tolerance and no silent exclusions.
"""
import json
import numpy as np
import mirt
from core.irt_python_mirt_parameters import python_mirt_binary_parameters


def run(n_persons=600, n_items=12, seeds=(2026, 2027)):
    results = []
    for model in ("1PL", "2PL", "3PL", "4PL"):
        for seed in seeds:
            simulated = mirt.simdata(model=model, n_persons=n_persons,
                                     n_items=n_items, seed=seed)
            fitted = mirt.fit_mirt(simulated, model=model)
            table = python_mirt_binary_parameters(fitted, model=model)
            converged = getattr(fitted, "converged", None)
            if callable(converged):
                converged = converged()
            estimates = table[["a", "b", "c", "d"]].to_numpy(float)
            results.append({
                "model": model, "seed": seed, "n_persons": n_persons,
                "n_items": n_items, "converged": (
                    bool(converged) if converged is not None else None),
                "finite_parameters": bool(np.isfinite(estimates).all()),
                "mean_discrimination": float(np.mean(estimates[:, 0])),
                "mean_difficulty": float(np.mean(estimates[:, 1])),
                "mean_guessing": float(np.mean(estimates[:, 2])),
                "mean_upper_asymptote": float(np.mean(estimates[:, 3])),
                "recovery_mae": None,
                "fit_success": bool(np.isfinite(estimates).all()),
                "recovery_note": (
                    "simdata true item parameters not available from this API call; "
                    "cannot calculate recovery MAE without documented ground truth"),
            })
    return {"engine": "Python mirt", "models": results,
            "status": "FIT_STABILITY_ONLY_NOT_PARAMETER_RECOVERY"}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
