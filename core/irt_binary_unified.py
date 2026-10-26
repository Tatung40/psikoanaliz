"""Single entry point for EserPsychometrics' five dichotomous IRT models.

Rasch is distinct from 1PL: Rasch fits a free latent SD; 1PL fixes SD=1.
Native estimators retain their established behavior; the optional Python mirt
engine is experimental and is never silently substituted after failure.
"""
from dataclasses import dataclass
import numpy as np
from core.irt import fit_irt
from core.rasch import fit_rasch
from core.irt_python_mirt_adapter import fit_python_mirt

BINARY_MODELS=("Rasch","1PL","2PL","3PL","4PL")

@dataclass
class BinaryFit:
    model: str
    engine: str
    result: object
    scientific_status: str
    warnings: tuple[str,...]

def fit_binary(data, model, *, engine="eser", **options):
    if model not in BINARY_MODELS: raise ValueError("Unsupported dichotomous model")
    if engine not in ("eser","python_mirt"): raise ValueError("Unknown engine")
    if engine=="python_mirt":
        if model=="Rasch":
            raise ValueError("Rasch free-latent-SD identification differs from Python mirt 1PL; use native Rasch")
        unknown=set(options)-{"max_iter","tol","n_quadpts"}
        if unknown:
            raise ValueError("Unsupported Python mirt options: "+", ".join(sorted(unknown)))
        fit=fit_python_mirt(data,model=model,**options)
        return BinaryFit(model,engine,fit,"EXPERIMENTAL_NOT_INDEPENDENTLY_VALIDATED",
                         ("Python mirt results: only execution smoke-tested; inferential claims withheld.",))
    if model=="Rasch":
        fit=fit_rasch(data,**options)
    else:
        x=np.asarray(data,float)
        if x.ndim!=2 or not np.isfinite(x).all():
            raise ValueError("Native 1PL–4PL require complete finite response matrices")
        fit=fit_irt(x,model=model,**options)
    return BinaryFit(model,"eser",fit,"NATIVE_MODEL_SPECIFIC_VALIDATION",
                     tuple(fit.get("warnings",())) if isinstance(fit,dict) else ())
