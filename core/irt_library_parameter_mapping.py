"""Strict, explicit mapping of externally estimated binary IRT parameters.

The library must provide documented a/b/c/d metrics. We intentionally do
not guess whether a table uses intercept, difficulty or multidimensional loadings.
"""
import numpy as np
import pandas as pd
from core.irt_library_curves import binary_curves

FIELDS = ("a", "b", "c", "d")


def normalize_binary_item_parameters(frame, *, model, item_names=None):
    if model not in ("1PL", "2PL", "3PL", "4PL"):
        raise ValueError("Only binary models are supported")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a documented parameter DataFrame")
    required = {"b"} if model == "1PL" else {"a", "b"}
    if model in ("3PL", "4PL"):
        required.add("c")
    if model == "4PL":
        required.add("d")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing explicit logistic-metric columns: " + ", ".join(sorted(missing)))
    n = len(frame)
    if n < 1:
        raise ValueError("Empty parameter table")
    a = (pd.to_numeric(frame["a"], errors="raise").to_numpy(float)
         if "a" in frame.columns else np.ones(n))
    b = pd.to_numeric(frame["b"], errors="raise").to_numpy(float)
    c = pd.to_numeric(frame["c"], errors="raise").to_numpy(float) if model in ("3PL", "4PL") else np.zeros(n)
    d = pd.to_numeric(frame["d"], errors="raise").to_numpy(float) if model == "4PL" else np.ones(n)
    if item_names is None:
        item_names = [f"M{i+1}" for i in range(n)]
    if len(item_names) != n or len(set(item_names)) != n:
        raise ValueError("Item names must match rows and be unique")
    binary_curves(np.array([0.]), a=a, b=b, c=c, d=d)
    return pd.DataFrame({"Madde": list(item_names), "a": a, "b": b, "c": c, "d": d})


def curves_from_normalized_table(frame, theta):
    if list(frame.columns) != ["Madde", "a", "b", "c", "d"]:
        raise ValueError("Expected normalized table with Madde,a,b,c,d")
    return binary_curves(theta, a=frame["a"], b=frame["b"], c=frame["c"], d=frame["d"])
