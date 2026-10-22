"""Hybrid IRT engine registry. Discovery is read-only; fitting is not activated.

The *Python* PyPI mirt package is distinct from R's mirt by Chalmers.
Engine support is advertised as candidate capabilities, not validated outcomes.
"""
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Optional

@dataclass(frozen=True)
class Engine:
    id: str
    import_name: Optional[str]
    models: tuple[str, ...]
    validated: tuple[str, ...] = ()
    optional: bool = True
    note: str = ""

ENGINES = {
    "eser": Engine("eser", "core.irt_mixed",
        ("1PL", "2PL", "3PL", "4PL", "MIXED_DICHOTOMOUS"),
        validated=(), optional=False,
        note="Legacy standalone routines remain unchanged; mixed inference experimental."),
    "python_mirt": Engine("python_mirt", "mirt",
        ("1PL", "2PL", "3PL", "4PL", "GRM", "PCM", "GPCM", "RSM",
         "NRM", "MIXED", "MIRT", "BIFACTOR", "TESTLET", "DIF"),
        note="Python mirt by Cameron Lyons, NOT R mirt. Candidate until independent verification."),
    "girth": Engine("girth", "girth",
        ("1PL", "2PL", "GRM", "PCM"),
        note="Legacy library; verify exact estimator/model support before enabling."),
    "py_irt": Engine("py_irt", "py_irt",
        ("1PL", "2PL", "4PL"),
        note="Optional Bayesian PyTorch/Pyro engine; not suitable as mandatory dependency."),
}

def engine_inventory():
    """Return honest install/validation status without importing heavy dependencies."""
    result=[]
    for engine in ENGINES.values():
        try:
            installed=bool(find_spec(engine.import_name)) if engine.import_name else False
        except (ImportError, ValueError, ModuleNotFoundError):
            installed=False
        result.append({
            "engine": engine.id,
            "installed": installed,
            "candidate_models": list(engine.models),
            "validated_models": list(engine.validated),
            "production_ready": False,
            "note": engine.note,
        })
    return result

def select_engine(model: str, preferred: str | None = None):
    """Fail closed: choose candidate, never silently substitute an unvalidated engine."""
    if preferred is not None:
        if preferred not in ENGINES:
            raise ValueError(f"Unknown engine: {preferred}")
        candidates=[ENGINES[preferred]]
    else:
        candidates=list(ENGINES.values())
    supported=[e for e in candidates if model in e.models]
    if not supported:
        raise ValueError(f"No registered engine for model {model}")
    raise RuntimeError(
        f"Model {model} is not production-authorized via hybrid registry. "
        "Complete independent benchmark and adapter tests before enabling."
    )
