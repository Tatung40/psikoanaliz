"""Isolated adapter for Cameron Lyons' Python mirt (NOT R mirt).

Primary external library calibration; raw fit preserved for model-specific inspection.
No claim that R parameter agreement or standard errors are validated.
"""
from dataclasses import dataclass, field
from numbers import Integral, Real
from importlib import import_module
import numpy as np

SUPPORTED = frozenset(("Rasch","1PL","2PL","3PL","4PL","GRM","PCM","GPCM","NRM"))
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
    estimation: str = "ML"
    solver_settings: dict = field(default_factory=dict)

ESTIMATIONS = ("ML", "MAP")


def solver_settings(*, max_iter=500, tol=1e-4, n_quadpts=21):
    """Exact public mirt.fit_mirt defaults; reject silent coercion/nonfinite values."""
    for name,value in (("max_iter",max_iter),("n_quadpts",n_quadpts)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(tol,(bool,np.bool_)) or not isinstance(tol,Real) or not np.isfinite(tol) or tol<=0:
        raise ValueError("tol must be finite and positive")
    return dict(max_iter=int(max_iter),tol=float(tol),n_quadpts=int(n_quadpts))


def fit_python_mirt(responses, *, model, n_categories=None, estimation="ML",
                    max_iter=500, tol=1e-4, n_quadpts=21):
    """One public fit call; no implicit engine fallback or unsupported inference.

    estimation="MAP" (3PL/4PL only) uses Python mirt's documented library default item priors
    (mirt.estimation.priors.default_priors: a~LogN(0,.5), b~N(0,2), c~Beta(2,8), d~Beta(8,2)).
    They were not tuned to any result; see validation/evidence/p1_3pl4pl for the pre-registered study.
    """
    settings=solver_settings(max_iter=max_iter,tol=tol,n_quadpts=n_quadpts)
    if estimation not in ESTIMATIONS:
        raise ValueError("estimation must be 'ML' or 'MAP'.")
    if estimation == "MAP" and model not in ("3PL", "4PL"):
        raise ValueError("MAP (library default priors) is enabled only for 3PL/4PL, where it was validated.")
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
    # Python Rasch: fixed a=1, N(0,1), unlike the separate native free-SD Rasch.
    kwargs={"model":"1PL" if model=="Rasch" else model,**settings}
    if cats is not None:kwargs["n_categories"]=cats
    if model=="NRM":
        # Python mirt 1.2.0 defect: se_method="auto" labels NRM SEs "oakes" but returns values ~0.45x the
        # empirical SD (95% CI coverage .62); an explicit "oakes" request gives ratio .99 / coverage .94.
        # Other families are identical under both settings. Evidence: validation/evidence/p4_polytomous.
        kwargs["se_method"]="oakes"
    if estimation=="MAP":
        from mirt.estimation.priors import default_priors
        kwargs["priors"]=default_priors(model)
    result=mirt.fit_mirt(x.astype(int),**kwargs)
    return PythonMirtResult("python_mirt",model,x.shape[0],x.shape[1],cats,result,estimation=estimation,solver_settings=settings)
