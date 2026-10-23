"""Verified Python mirt 1.2.0 coefficient schema to binary ICC/IIF tables.

The external fit.coef() columns were inspected in GitHub Actions against
actual fitted 1PL/2PL/3PL/4PL models. No legacy calibration is invoked.
"""
import pandas as pd
from core.irt_library_parameter_mapping import normalize_binary_item_parameters


def python_mirt_binary_parameters(fit, *, model, item_names=None):
    if model not in ("1PL", "2PL", "3PL", "4PL"):
        raise ValueError("Binary model required")
    coef = fit.coef()
    if not isinstance(coef, pd.DataFrame):
        raise TypeError("Python mirt 1.2.0 coef() must return a DataFrame")
    required = {"discrimination", "difficulty"}
    if model in ("3PL", "4PL"):
        required.add("guessing")
    if model == "4PL":
        required.add("upper")
    missing = required.difference(coef.columns)
    if missing:
        raise ValueError("Python mirt coefficient schema mismatch: " + ", ".join(sorted(missing)))
    raw = pd.DataFrame({"a": coef["discrimination"], "b": coef["difficulty"]})
    if model in ("3PL", "4PL"):
        raw["c"] = coef["guessing"]
    if model == "4PL":
        raw["d"] = coef["upper"]
    # Preserve the external 1PL common discrimination rather than assuming 1.
    return normalize_binary_item_parameters(raw, model=model, item_names=item_names)
