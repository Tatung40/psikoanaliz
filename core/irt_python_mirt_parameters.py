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


def coef_with_se_table(fit):
    """fit.coef_with_se() with SE of fixed/identification-constrained parameters blanked.

    Python mirt 1.2.0 reports SE = 0.0 for parameters that are not estimated (PCM/1PL a = 1,
    NRM category-1 slope/intercept = 0). An estimated parameter never has an exact zero SE,
    so 0.0 is shown as missing rather than as a perfectly precise estimate.
    """
    frame = fit.coef_with_se()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame = frame.copy()
    for column in [c for c in frame.columns if str(c).endswith("_se")]:
        frame[column] = frame[column].mask(frame[column] == 0)
    return frame
