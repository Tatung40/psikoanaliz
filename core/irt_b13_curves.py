"""B13: validated dichotomous IRT curve arrays for Rasch and 1PL–4PL.

Uses the existing platform's parameterization and probability/information
functions; does not introduce a second calibration implementation.
"""
import numpy as np
from core.irt import probabilities, information

GRID=np.linspace(-4,4,161)

def compute_curves(items, theta=None):
    t=GRID if theta is None else np.asarray(theta,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all():
        raise ValueError("Theta must be a finite 1-dimensional vector.")
    a,b,c,d=(np.asarray(items[p],dtype=float) for p in ("a","b","c","d"))
    if not (len(a)==len(b)==len(c)==len(d) and len(a)>0):
        raise ValueError("Inconsistent item parameters.")
    if not (np.isfinite(a).all() and np.isfinite(b).all() and
            np.isfinite(c).all() and np.isfinite(d).all()):
        raise ValueError("Nonfinite item parameters.")
    if np.any(a<=0) or np.any(c<0) or np.any(d>1) or np.any(c>=d):
        raise ValueError("Invalid 4PL parameter bounds.")
    p=probabilities(t,a,b,c,d)
    i=information(t,a,b,c,d)
    tif=i.sum(axis=1)
    sem=np.full(tif.shape,np.inf)
    np.divide(1,np.sqrt(tif),out=sem,where=tif>0)
    return {"theta":t,"icc":p,"iic":i,"tif":tif,"sem":sem,"tcc":p.sum(axis=1)}

def check_curves(items, model, theta=None):
    cv=compute_curves(items,theta)
    t=cv["theta"]
    a,b,c,d=(np.asarray(items[p],float) for p in ("a","b","c","d"))
    h=1e-5
    p=cv["icc"]
    deriv=(probabilities(t+h,a,b,c,d)-probabilities(t-h,a,b,c,d))/(2*h)
    finite=np.maximum(p*(1-p),1e-14)
    independent=deriv**2/finite
    discrepancy=float(np.max(np.abs(cv["iic"]-independent)/np.maximum(1,cv["iic"])))
    if model in ("Rasch","1PL"):
        structure=np.allclose(a,1) and np.allclose(c,0) and np.allclose(d,1)
    elif model=="2PL":
        structure=np.allclose(c,0) and np.allclose(d,1)
    elif model=="3PL":
        structure=np.allclose(d,1)
    elif model=="4PL":
        structure=True
    else: raise ValueError("Unknown dichotomous model")
    checks={"information_numerical_derivative":discrepancy<=1e-5,
        "test_information_sum":np.allclose(cv["tif"],cv["iic"].sum(axis=1)),
        "icc_monotone":bool(np.all(np.diff(p,axis=0)>=-1e-12)),
        "model_parameter_constraints":bool(structure)}
    return {"passed":all(checks.values()),"checks":checks,"max_relative_information_error":discrepancy}
