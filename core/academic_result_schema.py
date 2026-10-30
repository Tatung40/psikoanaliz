"""Versioned, strictly validated, JSON-safe academic result envelope.

Read-only transport layer: never recalculates or fabricates psychometric statistics.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Mapping

SCHEMA_VERSION = "1.0.0"
FAMILIES = frozenset({"CTT", "IRT", "GT"})
REQUIRED = ("schema_version", "family", "model", "sample", "parameters",
            "standard_errors", "fit_statistics", "tables", "warnings", "provenance")


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas-like scalars and tabular values without silent NaN coercion."""
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite result cannot be reported as a number")
        return value
    raise TypeError(f"Unsupported academic result value: {type(value).__name__}")


def build_result(*, family: str, model: str, n: int, k: int | None = None,
                 parameters=None, standard_errors=None, fit_statistics=None,
                 tables=None, warnings=None, provenance=None, metadata=None) -> dict:
    """Build a reproducible envelope from already-computed, observed outputs only."""
    if family not in FAMILIES:
        raise ValueError("family must be CTT, IRT or GT")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty name")
    if type(n) is not int or n < 1 or (k is not None and (type(k) is not int or k < 1)):
        raise ValueError("sample sizes must be positive integers")
    result = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "model": model.strip(),
        "sample": {"n": n, "k": k},
        "parameters": parameters if parameters is not None else [],
        "standard_errors": standard_errors if standard_errors is not None else [],
        "fit_statistics": fit_statistics if fit_statistics is not None else {},
        "tables": tables if tables is not None else {},
        "warnings": warnings if warnings is not None else [],
        "provenance": provenance if provenance is not None else {},
        "metadata": metadata if metadata is not None else {},
    }
    return validate_result(result)


def validate_result(result: Mapping) -> dict:
    if not isinstance(result, Mapping) or any(key not in result for key in REQUIRED):
        raise ValueError("Incomplete academic result schema")
    if result["schema_version"] != SCHEMA_VERSION or result["family"] not in FAMILIES:
        raise ValueError("Unknown schema version or family")
    if not isinstance(result["model"], str) or not result["model"].strip():
        raise ValueError("Invalid model name")
    sample = result["sample"]
    if not isinstance(sample, Mapping) or type(sample.get("n")) is not int or sample["n"] < 1:
        raise ValueError("Invalid sample size")
    if sample.get("k") is not None and (type(sample["k"]) is not int or sample["k"] < 1):
        raise ValueError("Invalid item count")
    for key in ("parameters", "standard_errors", "warnings"):
        if not isinstance(result[key], list):
            raise ValueError(f"{key} must be a list")
    for key in ("fit_statistics", "tables", "provenance"):
        if not isinstance(result[key], Mapping):
            raise ValueError(f"{key} must be a mapping")
    if "metadata" in result and not isinstance(result["metadata"], Mapping):
        raise ValueError("metadata must be a mapping")
    normalized = _json_safe(result)
    json.dumps(normalized, allow_nan=False, ensure_ascii=False)
    return normalized


def result_json(result: Mapping) -> str:
    return json.dumps(validate_result(result), ensure_ascii=False, allow_nan=False, indent=2)
