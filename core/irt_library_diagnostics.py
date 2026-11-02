"""Read-only display diagnostics from Python mirt FitResult.

No native engine inference or invented convergence metrics.
"""
import math


def library_itemfit(fit, responses, *, model):
    """Public itemfit API with the same scientific restrictions in UI and exports.

    Ordinal mean squares are descriptive; nominal item fit is unsupported here.
    S-X² p-values for eligible binary families are exploratory, not independently reference-validated.
    """
    import mirt
    import numpy as np
    from mirt.models.dichotomous import OneParameterLogistic
    if not bool(getattr(fit, "converged", False)):
        raise ValueError("Yakınsama sağlanmadı; madde uyumu hesaplanmadı.")
    binary = model in ("Rasch", "1PL", "2PL", "3PL", "4PL")
    if not binary and model not in ("GRM", "PCM", "GPCM"):
        raise ValueError("Bu model için madde uyumu desteklenmiyor.")
    statistics = ["infit", "outfit"] + (["S_X2"] if binary else [])
    frame = mirt.itemfit(fit, responses, statistics=statistics).copy()
    eligible = model == "2PL" or (model in ("Rasch", "1PL")
                                  and isinstance(getattr(fit, "model", None), OneParameterLogistic))
    if not eligible:
        # Exact public S-X² p-value names, including future adjustment requests.
        for column in ("p_value", "p_value_adjusted"):
            if column in frame:
                frame[column] = np.nan
    if not eligible:
        label = "DESCRIPTIVE / NOT CERTIFIED — S-X² çıkarımı kapalı"
    elif model == "2PL":
        # tests/test_psychometrics_reference.py: LSAT7 only, |ΔS-X²| <= .0022 vs R mirt 1.x.
        label = "EXPLORATORY / NOT CERTIFIED — 2PL S-X² yalnız LSAT7 örneğinde R mirt ile eşlendi"
    else:
        label = "EXPLORATORY / NOT CERTIFIED — kütüphane S-X² bağımsız referansla eşlenmedi"
    frame["Bilimsel durum"] = label
    return frame


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
