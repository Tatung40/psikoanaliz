"""Isolated adapter for Cameron Lyons' Python mirt (NOT R mirt).

Primary external library calibration; raw fit preserved for model-specific inspection.
No claim that R parameter agreement or standard errors are validated.
"""
from dataclasses import dataclass
from importlib import import_module
import numpy as np

SUPPORTED = frozenset(("1PL","2PL","3PL","4PL","GRM","PCM","GPCM","NRM"))
UNSUPPORTED_PUBLIC_MODELS = frozenset(("RSM",))
POLY = frozenset(("GRM","PCM","GPCM","RSM","NRM"))

@dataclass
class PythonMirtResult:
    engine: str
    model: str
    n_persons: int
    n_items: int
    n_categories: int | None
    fit: object
    scientific_status: str = "EXPERIMENTAL_NOT_INDEPENDENTLY_VALIDATED"

def fit_python_mirt(responses, *, model, n_categories=None):
    """One public fit call; no implicit engine fallback or unsupported inference."""
    if model not in SUPPORTED:
        raise ValueError(f"Unsupported model: {model}")
    if model in UNSUPPORTED_PUBLIC_MODELS:
        raise ValueError("Python mirt 1.2.0 fit_mirt API RSM modelini desteklemiyor. RSM için desteklenen bağımsız bir kütüphane motoru gereklidir.")
    try:
        x=np.asarray(responses, dtype=float)
    except (TypeError, ValueError) as ex:
        raise ValueError("Responses must be numeric.") from ex
    if x.ndim != 2 or min(x.shape)<3:
        raise ValueError("Expected persons x items, with at least three of each.")
    if not np.isfinite(x).all():
        raise ValueError("Missing/nonfinite responses require a validated missing-data policy.")
    if not np.array_equal(x, np.floor(x)) or np.any(x<0):
        raise ValueError("Responses must be nonnegative integer category codes.")
    if model not in POLY:
        if not np.isin(x,[0,1]).all():
            raise ValueError("Dichotomous models require 0/1 responses.")
        if n_categories is not None and n_categories!=2:
            raise ValueError("Dichotomous models have two categories.")
        cats=None
    else:
        if not isinstance(n_categories,int) or not 3<=n_categories<=20:
            raise ValueError("Polytomous n_categories must be 3..20.")
        if np.any(x>=n_categories):
            raise ValueError("Response exceeds declared category range.")
        if set(np.unique(x))!=set(range(n_categories)):
            raise ValueError("All categories must occur in the validation pilot.")
        counts = np.array([np.bincount(col.astype(int), minlength=n_categories)
                           for col in x.T])
        if np.any(counts == 0):
            raise ValueError("Her maddede beyan edilen tüm kategoriler gözlenmelidir; boş kategoriler sessizce yeniden kodlanmaz.")
        cats=n_categories
    # Constant items are not identifiable under ordinary IRT calibration.
    # Reject explicitly rather than returning misleading parameter estimates.
    if np.any(np.ptp(x, axis=0) == 0):
        bad = np.flatnonzero(np.ptp(x, axis=0) == 0) + 1
        raise ValueError(
            "Yanıt değişkenliği olmayan madde(ler) kestirilemez: "
            + ", ".join(map(str, bad.tolist()))
        )
    # Every dichotomous item must have observed successes and failures.
    # This check is separate from person-score extremes, which are allowed.
    try:
        mirt=import_module("mirt")
    except ImportError as ex:
        raise ImportError("Optional Python mirt missing: install mirt==1.2.0.") from ex
    kwargs={"model":model}
    if cats is not None:kwargs["n_categories"]=cats
    result=mirt.fit_mirt(x.astype(int),**kwargs)
    return PythonMirtResult("python_mirt",model,x.shape[0],x.shape[1],cats,result)
