"""Scientific engine-selection policy; no unvalidated fallback to experimental EM.

This module selects only engines explicitly certified for a model and data
condition. It does not infer certification from package installation or CI.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EngineDecision:
    engine: str | None
    status: str
    reason: str


def select_engine(model: str, *, has_missing: bool,
                  certified: Mapping[str, Mapping[str, bool]] | None = None
                  ) -> EngineDecision:
    """Select an explicitly certified engine; never silently impute or fall back.

    certified mapping: engine -> model -> bool, with optional key
    'MODEL:missing' for NaN-aware certification.
    """
    supported = {"1PL", "2PL", "3PL", "4PL", "GRM", "PCM", "GPCM", "NRM"}
    if model not in supported:
        raise ValueError(f"Unsupported IRT model: {model}")
    certified = certified or {}
    key = f"{model}:missing" if has_missing else model
    for engine in ("python_mirt", "other_python_library", "native"):
        if certified.get(engine, {}).get(key) is True:
            return EngineDecision(engine, "certified",
                                  f"{engine} independently certified for {key}")
    return EngineDecision(
        None, "blocked",
        f"No scientifically certified engine for {key}; "
        "do not report parameter estimates. Missing data are not "
        "automatically imputed or treated as incorrect responses."
    )
